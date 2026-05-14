from __future__ import annotations
import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import splu
from .domain import Domain
from .material import Material


class FlowState:
    """Snapshot of (u, v, p) on a Domain."""

    def __init__(self, u: np.ndarray, v: np.ndarray, p: np.ndarray, domain: Domain) -> None:
        self.u = u.copy()   # (nx+1, ny)
        self.v = v.copy()   # (nx, ny+1)
        self.p = p.copy()   # (nx, ny)
        self.domain = domain

    @property
    def u_cell(self) -> np.ndarray:
        """Cell-centre x-velocity (nx, ny)."""
        return 0.5 * (self.u[:-1, :] + self.u[1:, :])

    @property
    def v_cell(self) -> np.ndarray:
        """Cell-centre y-velocity (nx, ny)."""
        return 0.5 * (self.v[:, :-1] + self.v[:, 1:])

    @property
    def speed(self) -> np.ndarray:
        return np.sqrt(self.u_cell ** 2 + self.v_cell ** 2)

    @property
    def vorticity(self) -> np.ndarray:
        """Vorticity at cell corners (nx-1, ny-1), padded to (nx, ny)."""
        dx, dy = self.domain.dx, self.domain.dy
        # Natural location: corners (i, j) for i=1..nx-1, j=1..ny-1
        omega = ((self.v[1:, 1:-1] - self.v[:-1, 1:-1]) / dx
               - (self.u[1:-1, 1:] - self.u[1:-1, :-1]) / dy)   # (nx-1, ny-1)
        return np.pad(omega, ((0, 1), (0, 1)))                    # (nx, ny)

    def save(self, path: str) -> None:
        np.savez_compressed(path, u=self.u, v=self.v, p=self.p)

    @classmethod
    def load(cls, path: str, domain: Domain) -> FlowState:
        d = np.load(path)
        return cls(d['u'], d['v'], d['p'], domain)


class Solver:
    """
    2-D incompressible Navier-Stokes — Chorin projection on MAC staggered grid.

    Grid layout:
      u[i, j]  x-velocity at right face of cell (i-1, j)   shape (nx+1, ny)
      v[i, j]  y-velocity at top  face of cell (i, j-1)    shape (nx, ny+1)
      p[i, j]  pressure   at cell centre (i, j)            shape (nx, ny)

    Index 0 = leftmost/bottommost boundary face.
    """

    def __init__(self, domain: Domain, material: Material,
                 dt: float | None = None, relax: float = 1.0) -> None:
        self.domain   = domain
        self.material = material
        self.relax    = float(relax)   # pressure-correction under-relaxation (0 < relax ≤ 1)
        nx, ny = domain.nx, domain.ny

        self.u = np.zeros((nx + 1, ny),  dtype=np.float64)
        self.v = np.zeros((nx,     ny + 1), dtype=np.float64)
        self.p = np.zeros((nx,     ny),  dtype=np.float64)
        self._u_prev = self.u.copy()
        self._v_prev = self.v.copy()

        if dt is None:
            U_ref = max(abs(domain.bc_values.get('inlet_u', 0.0)),
                        abs(domain.bc_values.get('lid_u',   1.0)),
                        1e-6)
            dx_min = min(domain.dx, domain.dy)
            # 2-D CFL: combined u+v advection can reach ~3× U_ref in transient;
            # use 0.2 safety factor so dt*(u+v)/dx stays well below 1.
            dt_adv = 0.2 * dx_min / U_ref
            dt_vis = 0.5 * dx_min ** 2 / (4.0 * material.nu)
            dt = min(dt_adv, dt_vis)
        self.dt = float(dt)

        self._lu, self._rhs_bc, self._dirichlet_mask = self._build_poisson()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        duration: float,
        tol:          float = 1e-6,
        print_every:  int   = 500,
    ) -> FlowState:
        n_steps = max(1, int(duration / self.dt))
        for step in range(1, n_steps + 1):
            self.step()
            if step % print_every == 0 or step == n_steps:
                res = self._residual()
                print(f"  step {step:>6d}/{n_steps}  |du|/dt={res:.2e}")
                if res < tol and step > 50:
                    print(f"  Converged at step {step}.")
                    break
        return FlowState(self.u, self.v, self.p, self.domain)

    def step(self) -> None:
        self._u_prev[:] = self.u
        self._v_prev[:] = self.v

        u_star, v_star = self._advect_diffuse()
        self._apply_bc(u_star, v_star)
        self._mask_solid(u_star, v_star)

        p_new = self._solve_pressure(u_star, v_star)
        u_new, v_new = self._correct_velocity(u_star, v_star, p_new)
        self._apply_bc(u_new, v_new)
        self._mask_solid(u_new, v_new)

        self.u[:] = u_new
        self.v[:] = v_new
        self.p[:] = p_new

    # ------------------------------------------------------------------ #
    # Advection + diffusion
    # ------------------------------------------------------------------ #

    def _advect_diffuse(self):
        nx, ny = self.domain.nx, self.domain.ny
        dx, dy = self.domain.dx, self.domain.dy
        nu, dt = self.material.nu, self.dt
        u, v   = self.u, self.v     # (nx+1,ny), (nx,ny+1)

        u_star = u.copy()
        v_star = v.copy()

        # ---- Update u[1:-1, :]: interior x-faces (nx-1, ny) ----
        ui = u[1:-1, :]   # (nx-1, ny)

        # v interpolated to u-face positions (i=1..nx-1, j=0..ny-1)
        # v[k, j] is at x=(k+0.5)*dx; u-face at i*dx, (j+0.5)*dy
        # 4-point average: k=i-1,i  ×  j, j+1
        v_at_u = 0.25 * (v[:-1, :-1] + v[:-1, 1:] + v[1:, :-1] + v[1:, 1:])  # (nx-1, ny)

        # Upwind ∂u/∂x
        du_dx = np.where(ui > 0,
                         (u[1:-1, :] - u[:-2,  :]) / dx,
                         (u[2:,   :] - u[1:-1, :]) / dx)

        # Upwind ∂u/∂y  — pad u in y with edge values
        u_py = np.pad(ui, ((0, 0), (1, 1)), mode='edge')   # (nx-1, ny+2)
        du_dy = np.where(v_at_u > 0,
                         (ui           - u_py[:, :-2]) / dy,
                         (u_py[:, 2:]  - ui           ) / dy)

        # Diffusion ∇²u
        d2u_dx2 = (u[2:, :] - 2*ui + u[:-2, :]) / dx**2
        d2u_dy2 = (u_py[:, 2:] - 2*ui + u_py[:, :-2]) / dy**2

        u_star[1:-1, :] = ui + dt * (-ui * du_dx - v_at_u * du_dy
                                     + nu * (d2u_dx2 + d2u_dy2))

        # ---- Update v[:, 1:-1]: interior y-faces (nx, ny-1) ----
        vi = v[:, 1:-1]   # (nx, ny-1)

        # u interpolated to v-face positions (i=0..nx-1, j=1..ny-1)
        # u-face at i*dx, (j+0.5)*dy; v-face at (i+0.5)*dx, j*dy
        # 4-point average: i, i+1  ×  j-1, j
        u_at_v = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])  # (nx, ny-1)

        # Upwind ∂v/∂y
        dv_dy = np.where(vi > 0,
                         (v[:, 1:-1] - v[:, :-2]) / dy,
                         (v[:, 2:]   - v[:, 1:-1]) / dy)

        # Upwind ∂v/∂x — anti-symmetric ghost at no-slip walls (no-slip: v=0 at wall)
        # Neumann (edge) elsewhere (inlet/outlet).
        _bct = self.domain.bc_type
        v_left  = -vi[0:1,  :] if _bct.get('left')  == 'no_slip' else vi[0:1,  :]
        v_right = -vi[-1:,  :] if _bct.get('right') == 'no_slip' else vi[-1:,  :]
        v_px = np.concatenate([v_left, vi, v_right], axis=0)   # (nx+2, ny-1)
        dv_dx = np.where(u_at_v > 0,
                         (vi           - v_px[:-2, :]) / dx,
                         (v_px[2:, :]  - vi           ) / dx)

        # Diffusion ∇²v
        d2v_dy2 = (v[:, 2:] - 2*vi + v[:, :-2]) / dy**2
        d2v_dx2 = (v_px[2:, :] - 2*vi + v_px[:-2, :]) / dx**2

        v_star[:, 1:-1] = vi + dt * (-u_at_v * dv_dx - vi * dv_dy
                                     + nu * (d2v_dx2 + d2v_dy2))

        return u_star, v_star

    # ------------------------------------------------------------------ #
    # Boundary conditions
    # ------------------------------------------------------------------ #

    def _apply_bc(self, u: np.ndarray, v: np.ndarray) -> None:
        bct = self.domain.bc_type
        bcv = self.domain.bc_values
        u_in  = float(bcv.get('inlet_u', 0.0))
        v_in  = float(bcv.get('inlet_v', 0.0))
        lid_u = float(bcv.get('lid_u',   0.0))

        # Left (i=0 u-face)
        bt = bct.get('left', 'no_slip')
        if bt == 'inlet':
            u[0, :] = u_in
            v[0, 1:-1] = 0.0
        else:
            u[0, :] = 0.0

        # Right (i=nx u-face)
        bt = bct.get('right', 'no_slip')
        if bt == 'outlet':
            u[-1, :] = u[-2, :]        # zero-gradient
        else:
            u[-1, :] = 0.0

        # Bottom (j=0 v-face)
        bt = bct.get('bottom', 'no_slip')
        if bt == 'no_slip':
            v[:, 0] = 0.0
            u[1:-1, 0] = -u[1:-1, 1]   # no-slip ghost: u=0 at wall
        elif bt == 'lid':
            v[:, 0] = 0.0
            u[1:-1, 0] = 2.0 * lid_u - u[1:-1, 1]
        elif bt == 'inlet_v':
            v[:, 0] = v_in

        # Top (j=ny v-face)
        bt = bct.get('top', 'no_slip')
        if bt == 'no_slip':
            v[:, -1] = 0.0
            u[1:-1, -1] = -u[1:-1, -2]
        elif bt == 'lid':
            v[:, -1] = 0.0
            u[1:-1, -1] = 2.0 * lid_u - u[1:-1, -2]
        elif bt in ('outlet', 'outlet_v'):
            v[:, -1] = v[:, -2]

    def _mask_solid(self, u: np.ndarray, v: np.ndarray) -> None:
        solid = self.domain.solid    # (nx, ny) bool
        if not solid.any():
            return
        for i in range(self.domain.nx):
            mask = solid[i, :]
            if mask.any():
                u[i,   mask] = 0.0    # left x-face of solid cell
                u[i+1, mask] = 0.0    # right x-face
        for j in range(self.domain.ny):
            mask = solid[:, j]
            if mask.any():
                v[mask, j  ] = 0.0    # bottom y-face
                v[mask, j+1] = 0.0    # top y-face

    # ------------------------------------------------------------------ #
    # Pressure solver
    # ------------------------------------------------------------------ #

    def _solve_pressure(self, u_star: np.ndarray, v_star: np.ndarray) -> np.ndarray:
        nx, ny = self.domain.nx, self.domain.ny
        dx, dy = self.domain.dx, self.domain.dy
        rho, dt = self.material.rho, self.dt

        div = ((u_star[1:, :] - u_star[:-1, :]) / dx
             + (v_star[:, 1:] - v_star[:, :-1]) / dy)   # (nx, ny)
        rhs = (rho / dt) * div.flatten(order='F') + self._rhs_bc

        # Ghost-row cells (outlets + wall rows) are constrained by _build_poisson;
        # their RHS must be 0 (constraint equations, not Poisson source rows).
        rhs[self._dirichlet_mask] = 0.0
        rhs[self.domain.solid.flatten(order='F')] = 0.0

        p_flat = self._lu.solve(rhs)
        return p_flat.reshape((nx, ny), order='F')

    def _correct_velocity(self, u_star, v_star, p):
        dx, dy = self.domain.dx, self.domain.dy
        dt, rho = self.dt, self.material.rho
        α = self.relax   # under-relaxation factor (1.0 = no relaxation)

        u_new = u_star.copy()
        v_new = v_star.copy()
        u_new[1:-1, :] -= α * (dt / rho) * (p[1:, :] - p[:-1, :]) / dx
        v_new[:, 1:-1] -= α * (dt / rho) * (p[:, 1:] - p[:, :-1]) / dy
        return u_new, v_new

    # ------------------------------------------------------------------ #
    # Build sparse Laplacian (once at init)
    # ------------------------------------------------------------------ #

    def _build_poisson(self):
        """
        Build LU-factored sparse Laplacian for  ∇²p = rhs.

        Wall y-ghost rows (j=0 bottom, j=ny-1 top for no_slip/lid) are
        constrained to equal their neighbour: p[ghost] = p[real].
        This prevents ghost-row artifacts from driving spurious v corrections.

        Boundary conditions:
          Neumann (∂p/∂n = 0) at walls/inlets → reduces diagonal.
          Dirichlet (p = 0) at outlet face cells.
          Ghost-row constraint  p[j=0] = p[j=1]  (and top equivalent).
          Closed cavity: pin p=0 at one interior cell.
        """
        nx, ny = self.domain.nx, self.domain.ny
        dx, dy = self.domain.dx, self.domain.dy
        bct    = self.domain.bc_type
        n      = nx * ny

        _wall = ('no_slip', 'lid')
        ghost_bottom = bct.get('bottom') in _wall
        ghost_top    = bct.get('top')    in _wall

        outlet_right  = bct.get('right')  == 'outlet'
        outlet_top    = bct.get('top')    in ('outlet', 'outlet_v')
        outlet_left   = bct.get('left')   == 'outlet'
        outlet_bottom = bct.get('bottom') in ('outlet', 'outlet_v')
        has_outlet    = outlet_right or outlet_top or outlet_left or outlet_bottom

        def _is_outlet(i, j):
            if outlet_right  and i == nx - 1: return True
            if outlet_left   and i == 0:      return True
            if outlet_top    and j == ny - 1: return True
            if outlet_bottom and j == 0:      return True
            return False

        def _is_ghost(i, j):
            """Ghost rows: u-mirror rows at horizontal walls (treated as constraint cells)."""
            if ghost_bottom and j == 0:      return True
            if ghost_top    and j == ny - 1: return True
            return False

        # k(i,j) = i + j*nx   (Fortran / column-major order)
        L   = lil_matrix((n, n))
        rhs = np.zeros(n)

        for j in range(ny):
            for i in range(nx):
                k = i + j * nx

                if _is_outlet(i, j):
                    L[k, k] = 1.0
                    continue

                if _is_ghost(i, j):
                    # Constraint: p[ghost] = p[adjacent real row]
                    # → L p[ghost] - p[real] = 0
                    L[k, k] = 1.0
                    if j == 0:
                        L[k, i + 1 * nx] = -1.0   # p[j=0] = p[j=1]
                    else:
                        L[k, i + (ny-2) * nx] = -1.0  # p[j=ny-1] = p[j=ny-2]
                    continue

                c = 0.0

                # x-neighbours
                if i > 0:
                    if _is_outlet(i-1, j):
                        rhs[k] -= 0.0 / dx**2
                    else:
                        L[k, (i-1) + j*nx] += 1.0 / dx**2
                    c -= 1.0 / dx**2
                # else: Neumann at left wall

                if i < nx - 1:
                    if _is_outlet(i+1, j):
                        rhs[k] -= 0.0 / dx**2
                    else:
                        L[k, (i+1) + j*nx] += 1.0 / dx**2
                    c -= 1.0 / dx**2
                # else: Neumann at right wall

                # y-neighbours — ghost rows are coupled normally; the ghost
                # constraint rows (L p[ghost] = p[real]) handle p[ghost]=p[real].
                if j > 0:
                    if _is_outlet(i, j-1):
                        rhs[k] -= 0.0 / dy**2
                    else:
                        L[k, i + (j-1)*nx] += 1.0 / dy**2
                    c -= 1.0 / dy**2
                # else: Neumann at bottom

                if j < ny - 1:
                    if _is_outlet(i, j+1):
                        rhs[k] -= 0.0 / dy**2
                    else:
                        L[k, i + (j+1)*nx] += 1.0 / dy**2
                    c -= 1.0 / dy**2
                # else: Neumann at top

                L[k, k] = c if c != 0.0 else -1.0

        # Dirichlet mask: outlets + ghost rows (constraint rows)
        dirichlet_mask = np.zeros(n, dtype=bool)
        for j in range(ny):
            for i in range(nx):
                if _is_outlet(i, j) or _is_ghost(i, j):
                    dirichlet_mask[i + j * nx] = True

        # Closed cavity (no outlet): pin p=0 at one non-ghost interior cell
        if not has_outlet:
            # Find first non-ghost cell
            pin_k = None
            for j in range(ny):
                for i in range(nx):
                    if not _is_ghost(i, j):
                        pin_k = i + j * nx
                        break
                if pin_k is not None:
                    break
            if pin_k is None:
                pin_k = 0
            L[pin_k, :] = 0.0
            L[pin_k, pin_k] = 1.0
            rhs[pin_k]  = 0.0
            dirichlet_mask[pin_k] = True

        L = L.tocsc()
        lu = splu(L)
        return lu, rhs, dirichlet_mask

    def _residual(self) -> float:
        du = np.max(np.abs(self.u - self._u_prev)) / self.dt
        dv = np.max(np.abs(self.v - self._v_prev)) / self.dt
        return max(float(du), float(dv))
