from __future__ import annotations
import numpy as np
from scipy.sparse import eye as speye, kron, diags as spdiags
from scipy.sparse.linalg import splu
from .domain3d import Domain3D
from .material import Material


class FlowState3D:
    """Snapshot of (u, v, w, p) on a Domain3D."""

    def __init__(self, u: np.ndarray, v: np.ndarray,
                 w: np.ndarray, p: np.ndarray, domain: Domain3D) -> None:
        self.u = u.copy()   # (nx+1, ny, nz)
        self.v = v.copy()   # (nx, ny+1, nz)
        self.w = w.copy()   # (nx, ny, nz+1)
        self.p = p.copy()   # (nx, ny, nz)
        self.domain = domain

    @property
    def u_cell(self) -> np.ndarray:
        """Cell-centre x-velocity (nx, ny, nz)."""
        return 0.5 * (self.u[:-1, :, :] + self.u[1:, :, :])

    @property
    def v_cell(self) -> np.ndarray:
        return 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])

    @property
    def w_cell(self) -> np.ndarray:
        return 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])

    @property
    def speed(self) -> np.ndarray:
        return np.sqrt(self.u_cell**2 + self.v_cell**2 + self.w_cell**2)

    def save(self, path: str) -> None:
        np.savez_compressed(path, u=self.u, v=self.v, w=self.w, p=self.p)

    @classmethod
    def load(cls, path: str, domain: Domain3D) -> 'FlowState3D':
        d = np.load(path)
        return cls(d['u'], d['v'], d['w'], d['p'], domain)


class Solver3D:
    """
    3-D incompressible Navier-Stokes — Chorin projection on MAC staggered grid.

    Grid layout (column-major cell index k = i + j*nx + kz*nx*ny):
      u[i,j,k]  x-velocity at right  x-face of cell (i-1,j,k)   (nx+1, ny, nz)
      v[i,j,k]  y-velocity at top    y-face of cell (i,j-1,k)   (nx, ny+1, nz)
      w[i,j,k]  z-velocity at front  z-face of cell (i,j,k-1)   (nx, ny, nz+1)
      p[i,j,k]  pressure at cell centre (i,j,k)                  (nx, ny, nz)

    Ghost rows:
      j=0, j=ny-1 in u and w mirror adjacent real rows for top/bottom walls.
      k=0, k=nz-1 in u and v mirror adjacent real layers for front/back walls.

    Pressure solver:
      N ≤ 32768 (~32^3): pre-factored LU (splu), fast per-step.
      N  > 32768        : GMRES with diagonal preconditioner + warm start.
    """

    def __init__(
        self,
        domain:   Domain3D,
        material: Material,
        dt:       float | None = None,
        relax:    float = 1.0,
    ) -> None:
        self.domain   = domain
        self.material = material
        self.relax    = float(relax)
        nx, ny, nz = domain.nx, domain.ny, domain.nz

        self.u = np.zeros((nx + 1, ny,     nz),     dtype=np.float64)
        self.v = np.zeros((nx,     ny + 1, nz),     dtype=np.float64)
        self.w = np.zeros((nx,     ny,     nz + 1), dtype=np.float64)
        self.p = np.zeros((nx,     ny,     nz),     dtype=np.float64)
        self._u_prev = self.u.copy()
        self._v_prev = self.v.copy()
        self._w_prev = self.w.copy()

        if dt is None:
            U_ref = max(abs(domain.bc_values.get('inlet_u', 0.0)),
                        abs(domain.bc_values.get('lid_u',   1.0)),
                        1e-6)
            h = min(domain.dx, domain.dy, domain.dz)
            dt_adv = 0.15 * h / U_ref
            dt_vis = 0.5 * h**2 / (6.0 * material.nu)
            dt = min(dt_adv, dt_vis)
        self.dt = float(dt)

        N = nx * ny * nz
        self._use_direct = N <= 32768
        self._L, self._rhs_bc, self._dirichlet_mask = self._build_poisson()

        if self._use_direct:
            self._lu = splu(self._L)
        else:
            diag = np.abs(np.array(self._L.diagonal()))
            diag[diag < 1e-14] = 1.0
            self._precond_diag = 1.0 / diag

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        duration:    float,
        tol:         float = 1e-6,
        print_every: int   = 200,
    ) -> FlowState3D:
        n_steps = max(1, int(duration / self.dt))
        for step in range(1, n_steps + 1):
            self.step()
            if step % print_every == 0 or step == n_steps:
                res = self._residual()
                print(f"  step {step:>6d}/{n_steps}  |du|/dt={res:.2e}")
                if res < tol and step > 50:
                    print(f"  Converged at step {step}.")
                    break
        return FlowState3D(self.u, self.v, self.w, self.p, self.domain)

    def step(self) -> None:
        self._u_prev[:] = self.u
        self._v_prev[:] = self.v
        self._w_prev[:] = self.w

        u_star, v_star, w_star = self._advect_diffuse()
        self._apply_bc(u_star, v_star, w_star)
        self._mask_solid(u_star, v_star, w_star)

        p_new = self._solve_pressure(u_star, v_star, w_star)
        u_new, v_new, w_new = self._correct_velocity(u_star, v_star, w_star, p_new)
        self._apply_bc(u_new, v_new, w_new)
        self._mask_solid(u_new, v_new, w_new)

        self.u[:] = u_new
        self.v[:] = v_new
        self.w[:] = w_new
        self.p[:] = p_new

    # ------------------------------------------------------------------ #
    # Advection + diffusion  (first-order upwind advection, central diffusion)
    # ------------------------------------------------------------------ #

    def _advect_diffuse(self):
        nx, ny, nz = self.domain.nx, self.domain.ny, self.domain.nz
        dx, dy, dz = self.domain.dx, self.domain.dy, self.domain.dz
        nu, dt = self.material.nu, self.dt
        u, v, w = self.u, self.v, self.w
        bct = self.domain.bc_type

        u_star = u.copy()
        v_star = v.copy()
        w_star = w.copy()

        # ---- u[1:-1, :, :]: interior x-faces  (nx-1, ny, nz) ----
        ui = u[1:-1, :, :]

        # v interpolated to u-face: avg over x(i-1,i) and y(j,j+1)
        v_at_u = 0.25 * (v[:-1, :-1, :] + v[:-1, 1:, :] +
                         v[1:,  :-1, :] + v[1:,  1:, :])  # (nx-1, ny, nz)

        # w interpolated to u-face: avg over x(i-1,i) and z(k,k+1)
        w_at_u = 0.25 * (w[:-1, :, :-1] + w[:-1, :, 1:] +
                         w[1:,  :, :-1] + w[1:,  :, 1:])  # (nx-1, ny, nz)

        # upwind ∂u/∂x
        du_dx = np.where(ui > 0,
                         (u[1:-1, :, :] - u[:-2, :, :]) / dx,
                         (u[2:,   :, :] - u[1:-1, :, :]) / dx)

        # upwind ∂u/∂y — edge-pad in y (uses ghost rows j=0,ny-1 set by BC)
        u_py = np.pad(ui, ((0, 0), (1, 1), (0, 0)), mode='edge')
        du_dy = np.where(v_at_u > 0,
                         (ui - u_py[:, :-2, :]) / dy,
                         (u_py[:, 2:, :] - ui) / dy)

        # upwind ∂u/∂z — edge-pad in z (uses ghost layers k=0,nz-1)
        u_pz = np.pad(ui, ((0, 0), (0, 0), (1, 1)), mode='edge')
        du_dz = np.where(w_at_u > 0,
                         (ui - u_pz[:, :, :-2]) / dz,
                         (u_pz[:, :, 2:] - ui) / dz)

        d2u_dx2 = (u[2:,   :, :] - 2*ui + u[:-2, :, :]) / dx**2
        d2u_dy2 = (u_py[:, 2:, :] - 2*ui + u_py[:, :-2, :]) / dy**2
        d2u_dz2 = (u_pz[:, :, 2:] - 2*ui + u_pz[:, :, :-2]) / dz**2

        u_star[1:-1, :, :] = ui + dt * (
            -ui * du_dx - v_at_u * du_dy - w_at_u * du_dz
            + nu * (d2u_dx2 + d2u_dy2 + d2u_dz2))

        # ---- v[:, 1:-1, :]: interior y-faces  (nx, ny-1, nz) ----
        vi = v[:, 1:-1, :]

        # u interpolated to v-face: avg over x(i,i+1) and y(j-1,j)
        u_at_v = 0.25 * (u[:-1, :-1, :] + u[:-1, 1:, :] +
                         u[1:,  :-1, :] + u[1:,  1:, :])  # (nx, ny-1, nz)

        # w interpolated to v-face: avg over y(j-1,j) and z(k,k+1)
        w_at_v = 0.25 * (w[:, :-1, :-1] + w[:, :-1, 1:] +
                         w[:, 1:,  :-1] + w[:, 1:,  1:])  # (nx, ny-1, nz)

        # upwind ∂v/∂y
        dv_dy = np.where(vi > 0,
                         (v[:, 1:-1, :] - v[:, :-2, :]) / dy,
                         (v[:, 2:,   :] - v[:, 1:-1, :]) / dy)

        # upwind ∂v/∂x — anti-symmetric ghost at no-slip left/right walls
        v_left  = (-vi[0:1,  :, :] if bct.get('left')  == 'no_slip'
                   else vi[0:1, :, :])
        v_right = (-vi[-1:,  :, :] if bct.get('right') == 'no_slip'
                   else vi[-1:, :, :])
        v_px = np.concatenate([v_left, vi, v_right], axis=0)  # (nx+2, ny-1, nz)
        dv_dx = np.where(u_at_v > 0,
                         (vi - v_px[:-2, :, :]) / dx,
                         (v_px[2:, :, :] - vi) / dx)

        # upwind ∂v/∂z — anti-symmetric ghost at no-slip front/back walls
        v_front = (-vi[:, :, 0:1] if bct.get('front') == 'no_slip'
                   else vi[:, :, 0:1])
        v_back  = (-vi[:, :, -1:] if bct.get('back')  == 'no_slip'
                   else vi[:, :, -1:])
        v_pz = np.concatenate([v_front, vi, v_back], axis=2)  # (nx, ny-1, nz+2)
        dv_dz = np.where(w_at_v > 0,
                         (vi - v_pz[:, :, :-2]) / dz,
                         (v_pz[:, :, 2:] - vi) / dz)

        d2v_dy2 = (v[:, 2:, :] - 2*vi + v[:, :-2, :]) / dy**2
        d2v_dx2 = (v_px[2:, :, :] - 2*vi + v_px[:-2, :, :]) / dx**2
        d2v_dz2 = (v_pz[:, :, 2:] - 2*vi + v_pz[:, :, :-2]) / dz**2

        v_star[:, 1:-1, :] = vi + dt * (
            -u_at_v * dv_dx - vi * dv_dy - w_at_v * dv_dz
            + nu * (d2v_dx2 + d2v_dy2 + d2v_dz2))

        # ---- w[:, :, 1:-1]: interior z-faces  (nx, ny, nz-1) ----
        wi = w[:, :, 1:-1]

        # u interpolated to w-face: avg over x(i,i+1) and z(k-1,k)
        u_at_w = 0.25 * (u[:-1, :, :-1] + u[:-1, :, 1:] +
                         u[1:,  :, :-1] + u[1:,  :, 1:])  # (nx, ny, nz-1)

        # v interpolated to w-face: avg over y(j,j+1) and z(k-1,k)
        v_at_w = 0.25 * (v[:, :-1, :-1] + v[:, :-1, 1:] +
                         v[:, 1:,  :-1] + v[:, 1:,  1:])  # (nx, ny, nz-1)

        # upwind ∂w/∂z
        dw_dz = np.where(wi > 0,
                         (w[:, :, 1:-1] - w[:, :, :-2]) / dz,
                         (w[:, :, 2:]   - w[:, :, 1:-1]) / dz)

        # upwind ∂w/∂x — anti-symmetric ghost at no-slip left/right walls
        w_left  = (-wi[0:1, :, :] if bct.get('left')  == 'no_slip'
                   else wi[0:1, :, :])
        w_right = (-wi[-1:, :, :] if bct.get('right') == 'no_slip'
                   else wi[-1:, :, :])
        w_px = np.concatenate([w_left, wi, w_right], axis=0)  # (nx+2, ny, nz-1)
        dw_dx = np.where(u_at_w > 0,
                         (wi - w_px[:-2, :, :]) / dx,
                         (w_px[2:, :, :] - wi) / dx)

        # upwind ∂w/∂y — edge-pad in y (uses ghost y-rows in wi)
        w_py = np.pad(wi, ((0, 0), (1, 1), (0, 0)), mode='edge')
        dw_dy = np.where(v_at_w > 0,
                         (wi - w_py[:, :-2, :]) / dy,
                         (w_py[:, 2:, :] - wi) / dy)

        d2w_dz2 = (w[:, :, 2:] - 2*wi + w[:, :, :-2]) / dz**2
        d2w_dx2 = (w_px[2:, :, :] - 2*wi + w_px[:-2, :, :]) / dx**2
        d2w_dy2 = (w_py[:, 2:, :] - 2*wi + w_py[:, :-2, :]) / dy**2

        w_star[:, :, 1:-1] = wi + dt * (
            -u_at_w * dw_dx - v_at_w * dw_dy - wi * dw_dz
            + nu * (d2w_dx2 + d2w_dy2 + d2w_dz2))

        return u_star, v_star, w_star

    # ------------------------------------------------------------------ #
    # Boundary conditions
    # ------------------------------------------------------------------ #

    def _apply_bc(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> None:
        bct = self.domain.bc_type
        bcv = self.domain.bc_values
        u_in  = float(bcv.get('inlet_u', 0.0))
        lid_u = float(bcv.get('lid_u',   0.0))

        # Left face (x=0)
        bt = bct.get('left', 'no_slip')
        if bt == 'inlet':
            u[0, :, :] = u_in
            v[0, :, :] = 0.0
            w[0, :, :] = 0.0
        else:
            u[0, :, :] = 0.0

        # Right face (x=nx)
        bt = bct.get('right', 'no_slip')
        if bt == 'outlet':
            u[-1, :, :] = u[-2, :, :]
        else:
            u[-1, :, :] = 0.0

        # Bottom wall (y=0): v-face is ON the wall; u and w need ghost y-rows
        bt = bct.get('bottom', 'no_slip')
        if bt in ('no_slip',):
            v[:, 0, :] = 0.0
            u[1:-1, 0, :] = -u[1:-1, 1, :]
            w[:,    0, 1:-1] = -w[:,    1, 1:-1]
        elif bt == 'lid':
            v[:, 0, :] = 0.0
            u[1:-1, 0, :] = 2.0 * lid_u - u[1:-1, 1, :]
            w[:,    0, 1:-1] = -w[:,    1, 1:-1]

        # Top wall (y=ny)
        bt = bct.get('top', 'no_slip')
        if bt == 'no_slip':
            v[:, -1, :] = 0.0
            u[1:-1, -1, :] = -u[1:-1, -2, :]
            w[:,    -1, 1:-1] = -w[:,    -2, 1:-1]
        elif bt == 'lid':
            v[:, -1, :] = 0.0
            u[1:-1, -1, :] = 2.0 * lid_u - u[1:-1, -2, :]
            w[:,    -1, 1:-1] = -w[:,    -2, 1:-1]
        elif bt in ('outlet', 'outlet_v'):
            v[:, -1, :] = v[:, -2, :]

        # Front wall (z=0): w-face is ON the wall; u and v need ghost z-layers
        bt = bct.get('front', 'no_slip')
        if bt == 'no_slip':
            w[:, :, 0] = 0.0
            u[1:-1, :, 0] = -u[1:-1, :, 1]
            v[:, 1:-1, 0] = -v[:, 1:-1, 1]

        # Back wall (z=nz)
        bt = bct.get('back', 'no_slip')
        if bt == 'no_slip':
            w[:, :, -1] = 0.0
            u[1:-1, :, -1] = -u[1:-1, :, -2]
            v[:, 1:-1, -1] = -v[:, 1:-1, -2]

    def _mask_solid(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> None:
        solid = self.domain.solid
        if not solid.any():
            return
        ix, iy, iz = np.where(solid)
        for i, j, k in zip(ix.tolist(), iy.tolist(), iz.tolist()):
            u[i,   j, k] = 0.0;  u[i+1, j, k] = 0.0
            v[i, j,   k] = 0.0;  v[i, j+1, k] = 0.0
            w[i, j, k]   = 0.0;  w[i, j, k+1] = 0.0

    # ------------------------------------------------------------------ #
    # Pressure solver
    # ------------------------------------------------------------------ #

    def _solve_pressure(self, u_star, v_star, w_star) -> np.ndarray:
        nx, ny, nz = self.domain.nx, self.domain.ny, self.domain.nz
        dx, dy, dz = self.domain.dx, self.domain.dy, self.domain.dz
        rho, dt = self.material.rho, self.dt

        div = ((u_star[1:, :, :] - u_star[:-1, :, :]) / dx
             + (v_star[:, 1:, :] - v_star[:, :-1, :]) / dy
             + (w_star[:, :, 1:] - w_star[:, :, :-1]) / dz)  # (nx, ny, nz)

        rhs = (rho / dt) * div.ravel(order='F') + self._rhs_bc
        rhs[self._dirichlet_mask] = 0.0
        rhs[self.domain.solid.ravel(order='F')] = 0.0

        if self._use_direct:
            p_flat = self._lu.solve(rhs)
        else:
            from scipy.sparse.linalg import gmres, LinearOperator
            x0 = self.p.ravel(order='F')
            M  = LinearOperator((rhs.size, rhs.size),
                                matvec=lambda x: self._precond_diag * x)
            p_flat, info = gmres(self._L, rhs, x0=x0, M=M,
                                 tol=1e-7, restart=50, maxiter=300)
            if info != 0:
                p_flat = x0

        return p_flat.reshape((nx, ny, nz), order='F')

    def _correct_velocity(self, u_star, v_star, w_star, p):
        dx, dy, dz = self.domain.dx, self.domain.dy, self.domain.dz
        dt, rho = self.dt, self.material.rho
        α = self.relax

        u_new = u_star.copy()
        v_new = v_star.copy()
        w_new = w_star.copy()
        u_new[1:-1, :, :] -= α * (dt / rho) * (p[1:,  :,  :] - p[:-1, :,  :]) / dx
        v_new[:,  1:-1, :] -= α * (dt / rho) * (p[:,  1:,  :] - p[:,  :-1, :]) / dy
        w_new[:,  :, 1:-1] -= α * (dt / rho) * (p[:,  :,  1:] - p[:,  :, :-1]) / dz
        return u_new, v_new, w_new

    # ------------------------------------------------------------------ #
    # Build sparse 3-D Laplacian (once at init)
    # ------------------------------------------------------------------ #

    def _build_poisson(self):
        """
        Build 3-D Laplacian for ∇²p = rhs using Kronecker products.

        Column-major ordering: k = i + j*nx + kz*nx*ny.

        Ghost cells (j=0, j=ny-1 for walls; k=0, k=nz-1 for walls) are
        handled by the Neumann boundary in the 1-D Laplacian plus zeroing
        their RHS contribution — this enforces p[ghost] = p[adjacent].

        Outlet cells and solid cells get identity rows (Dirichlet p=0).
        Closed cavity: one cell pinned to p=0.
        """
        nx, ny, nz = self.domain.nx, self.domain.ny, self.domain.nz
        dx, dy, dz = self.domain.dx, self.domain.dy, self.domain.dz
        bct = self.domain.bc_type
        N   = nx * ny * nz

        _wall = ('no_slip', 'lid')
        ghost_j0  = bct.get('bottom') in _wall
        ghost_jny = bct.get('top')    in _wall
        ghost_k0  = bct.get('front')  in _wall
        ghost_knz = bct.get('back')   in _wall

        outlet_right  = bct.get('right')  == 'outlet'
        outlet_left   = bct.get('left')   == 'outlet'
        outlet_top    = bct.get('top')    in ('outlet', 'outlet_v')
        outlet_bottom = bct.get('bottom') in ('outlet', 'outlet_v')
        has_outlet    = outlet_right or outlet_left or outlet_top or outlet_bottom

        # ---- 1-D Neumann Laplacians ----
        def _neumann_lap(n, h):
            d = np.full(n, -2.0 / h**2)
            d[0] = d[-1] = -1.0 / h**2
            off = np.ones(n - 1) / h**2
            return spdiags([off, d, off], [-1, 0, 1], shape=(n, n), format='csr')

        Tx = _neumann_lap(nx, dx)
        Ty = _neumann_lap(ny, dy)
        Tz = _neumann_lap(nz, dz)
        Ix = speye(nx, format='csr')
        Iy = speye(ny, format='csr')
        Iz = speye(nz, format='csr')

        # Full 3-D Laplacian (column-major: i fastest, then j, then kz)
        L = (kron(kron(Iz, Iy), Tx, format='csr') +
             kron(kron(Iz, Ty), Ix, format='csr') +
             kron(kron(Tz, Iy), Ix, format='csr'))
        L = L.tolil()

        rhs = np.zeros(N)

        # ---- Coordinate arrays for vectorised indexing ----
        Ia, Ja, Ka = np.mgrid[0:nx, 0:ny, 0:nz]
        Ia_f = Ia.ravel(); Ja_f = Ja.ravel(); Ka_f = Ka.ravel()
        flat  = (Ia_f + Ja_f * nx + Ka_f * nx * ny).astype(np.intp)

        dirichlet_mask = np.zeros(N, dtype=bool)

        # ---- Ghost cells: mark RHS = 0 (Neumann kron handles the constraint) ----
        if ghost_j0:  dirichlet_mask[flat[Ja_f == 0]]     = True
        if ghost_jny: dirichlet_mask[flat[Ja_f == ny - 1]] = True
        if ghost_k0:  dirichlet_mask[flat[Ka_f == 0]]     = True
        if ghost_knz: dirichlet_mask[flat[Ka_f == nz - 1]] = True

        def _set_dirichlet(idx_arr):
            for k in idx_arr.tolist():
                L.rows[k] = [k]
                L.data[k] = [1.0]
            dirichlet_mask[idx_arr] = True

        # ---- Outlet faces: Dirichlet p = 0 ----
        if outlet_right:  _set_dirichlet(flat[Ia_f == nx - 1])
        if outlet_left:   _set_dirichlet(flat[Ia_f == 0])
        if outlet_top:    _set_dirichlet(flat[Ja_f == ny - 1])
        if outlet_bottom: _set_dirichlet(flat[Ja_f == 0])

        # ---- Solid cells: Dirichlet p = 0 ----
        solid_flat = self.domain.solid.ravel(order='F')
        solid_idx  = flat[solid_flat]
        if solid_idx.size > 0:
            _set_dirichlet(solid_idx)

        # ---- Closed cavity: pin one free cell to p = 0 ----
        if not has_outlet:
            free = flat[~dirichlet_mask]
            if free.size > 0:
                pin_k = int(free[0])
                L.rows[pin_k] = [pin_k]
                L.data[pin_k] = [1.0]
                rhs[pin_k] = 0.0
                dirichlet_mask[pin_k] = True

        L_csc = L.tocsc()
        return L_csc, rhs, dirichlet_mask

    # ------------------------------------------------------------------ #
    def _residual(self) -> float:
        du = np.max(np.abs(self.u - self._u_prev)) / self.dt
        dv = np.max(np.abs(self.v - self._v_prev)) / self.dt
        dw = np.max(np.abs(self.w - self._w_prev)) / self.dt
        return max(float(du), float(dv), float(dw))
