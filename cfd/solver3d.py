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

    Grid layout (column-major cell index k = i + j*nx + kz*nx*ny);
    all nx × ny × nz pressure cells are real fluid cells:
      u[i,j,k]  x-velocity at right  x-face of cell (i-1,j,k)   (nx+1, ny, nz)
      v[i,j,k]  y-velocity at top    y-face of cell (i,j-1,k)   (nx, ny+1, nz)
      w[i,j,k]  z-velocity at front  z-face of cell (i,j,k-1)   (nx, ny, nz+1)
      p[i,j,k]  pressure at cell centre (i,j,k)                  (nx, ny, nz)

    Walls live on the boundary faces (u at i=0,nx; v at j=0,ny; w at k=0,nz)
    of the unit-aligned domain [0,Lx]×[0,Ly]×[0,Lz].  Stencil ghosts for the
    interior u/v/w-faces that need values one cell beyond a wall are built
    inline in _advect_diffuse from the BC type, not stored in the arrays.

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

        self._precompute_spacings()

        if dt is None:
            U_ref = max(abs(domain.bc_values.get('inlet_u', 0.0)),
                        abs(domain.bc_values.get('lid_u',   1.0)),
                        1e-6)
            # Smallest cell in any direction sets the advection / viscous CFL
            # on a stretched grid.
            h = min(float(domain.dx_arr.min()),
                    float(domain.dy_arr.min()),
                    float(domain.dz_arr.min()))
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
    # Spacing caches — derived from domain.dx_arr / dy_arr / dz_arr.  On a
    # uniform grid every entry collapses to the constant dx / dy / dz.
    # ------------------------------------------------------------------ #

    def _precompute_spacings(self) -> None:
        dx_arr = self.domain.dx_arr
        dy_arr = self.domain.dy_arr
        dz_arr = self.domain.dz_arr
        nx, ny, nz = self.domain.nx, self.domain.ny, self.domain.nz

        def _face_dist(spacing):
            """Cell-centre-to-centre distances (n+1,), incl. wall-image rows."""
            n = spacing.size
            d         = np.empty(n + 1)
            d[0]      = spacing[0]                           # ghost ↔ first
            d[n]      = spacing[n - 1]                       # last ↔ ghost
            d[1:n]    = 0.5 * (spacing[:-1] + spacing[1:])
            return d

        dy_uy = _face_dist(dy_arr)
        dz_uz = _face_dist(dz_arr)
        dx_vx = _face_dist(dx_arr)

        # --- u-array stencil distances ---
        # ui[k, j, l] = u-face[k+1, j, l].
        self._bdist_x_u = dx_arr[:-1, None, None]              # (nx-1, 1, 1)
        self._fdist_x_u = dx_arr[1:,  None, None]
        self._bdist_y_u = dy_uy[:-1][None, :, None]            # (1, ny, 1)
        self._fdist_y_u = dy_uy[1:][None,  :, None]
        self._bdist_z_u = dz_uz[:-1][None, None, :]            # (1, 1, nz)
        self._fdist_z_u = dz_uz[1:][None,  None, :]

        # --- v-array stencil distances ---
        self._bdist_y_v = dy_arr[:-1][None, :, None]           # (1, ny-1, 1)
        self._fdist_y_v = dy_arr[1:][None,  :, None]
        self._bdist_x_v = dx_vx[:-1][:, None, None]            # (nx, 1, 1)
        self._fdist_x_v = dx_vx[1:][:,  None, None]
        self._bdist_z_v = dz_uz[:-1][None, None, :]            # (1, 1, nz)
        self._fdist_z_v = dz_uz[1:][None,  None, :]

        # --- w-array stencil distances ---
        self._bdist_z_w = dz_arr[:-1][None, None, :]           # (1, 1, nz-1)
        self._fdist_z_w = dz_arr[1:][None,  None, :]
        self._bdist_x_w = dx_vx[:-1][:, None, None]            # (nx, 1, 1)
        self._fdist_x_w = dx_vx[1:][:,  None, None]
        self._bdist_y_w = dy_uy[:-1][None, :, None]            # (1, ny, 1)
        self._fdist_y_w = dy_uy[1:][None,  :, None]

        # --- divergence + pressure-gradient distances ---
        self._inv_dx_cell = (1.0 / dx_arr)[:, None, None]      # (nx, 1, 1)
        self._inv_dy_cell = (1.0 / dy_arr)[None, :, None]
        self._inv_dz_cell = (1.0 / dz_arr)[None, None, :]
        self._dx_pcorr    = (0.5 * (dx_arr[:-1] + dx_arr[1:]))[:, None, None]
        self._dy_pcorr    = (0.5 * (dy_arr[:-1] + dy_arr[1:]))[None, :, None]
        self._dz_pcorr    = (0.5 * (dz_arr[:-1] + dz_arr[1:]))[None, None, :]

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
        nu, dt = self.material.nu, self.dt
        u, v, w = self.u, self.v, self.w
        bct    = self.domain.bc_type
        lid_u  = float(self.domain.bc_values.get('lid_u', 0.0))

        bdx_u, fdx_u = self._bdist_x_u, self._fdist_x_u
        bdy_u, fdy_u = self._bdist_y_u, self._fdist_y_u
        bdz_u, fdz_u = self._bdist_z_u, self._fdist_z_u
        bdx_v, fdx_v = self._bdist_x_v, self._fdist_x_v
        bdy_v, fdy_v = self._bdist_y_v, self._fdist_y_v
        bdz_v, fdz_v = self._bdist_z_v, self._fdist_z_v
        bdx_w, fdx_w = self._bdist_x_w, self._fdist_x_w
        bdy_w, fdy_w = self._bdist_y_w, self._fdist_y_w
        bdz_w, fdz_w = self._bdist_z_w, self._fdist_z_w

        u_star = u.copy()
        v_star = v.copy()
        w_star = w.copy()

        # Map (variable, axis) → (low-wall name, high-wall name).  Each interior
        # face array needs an image one cell beyond the two walls parallel to
        # the stencil axis:
        #   no_slip → anti-symmetric image (wall velocity = 0)
        #   lid     → 2 lid_u − image  (wall x-velocity = lid_u; u only)
        #   else    → zero-gradient (outlet / inlet / free-slip)
        WALL_OF = {
            ('u', 1): ('bottom', 'top'),   ('u', 2): ('front', 'back'),
            ('v', 0): ('left',   'right'), ('v', 2): ('front', 'back'),
            ('w', 0): ('left',   'right'), ('w', 1): ('bottom', 'top'),
        }

        def _pad(arr, var, axis, allow_lid):
            low_name, high_name = WALL_OF[(var, axis)]
            lo_ref = np.take(arr, [0],                   axis=axis)
            hi_ref = np.take(arr, [arr.shape[axis] - 1], axis=axis)

            def _image(ref, name):
                bt = bct.get(name, 'no_slip')
                if bt == 'no_slip':              return -ref
                if bt == 'lid' and allow_lid:    return 2.0 * lid_u - ref
                return ref                       # outlet / inlet / free-slip

            return np.concatenate(
                [_image(lo_ref, low_name), arr, _image(hi_ref, high_name)],
                axis=axis,
            )

        # ---- u[1:-1, :, :]: interior x-faces  (nx-1, ny, nz) ----
        ui = u[1:-1, :, :]

        v_at_u = 0.25 * (v[:-1, :-1, :] + v[:-1, 1:, :] +
                         v[1:,  :-1, :] + v[1:,  1:, :])
        w_at_u = 0.25 * (w[:-1, :, :-1] + w[:-1, :, 1:] +
                         w[1:,  :, :-1] + w[1:,  :, 1:])

        du_dx = np.where(ui > 0,
                         (u[1:-1, :, :] - u[:-2, :, :]) / bdx_u,
                         (u[2:,   :, :] - u[1:-1, :, :]) / fdx_u)

        u_py = _pad(ui, var='u', axis=1, allow_lid=True)
        du_dy = np.where(v_at_u > 0,
                         (ui             - u_py[:, :-2, :]) / bdy_u,
                         (u_py[:, 2:, :] - ui             ) / fdy_u)

        u_pz = _pad(ui, var='u', axis=2, allow_lid=False)
        du_dz = np.where(w_at_u > 0,
                         (ui             - u_pz[:, :, :-2]) / bdz_u,
                         (u_pz[:, :, 2:] - ui             ) / fdz_u)

        # Non-uniform 2nd derivative: 2/(h_L+h_R) · (Δ_R/h_R − Δ_L/h_L)
        d2u_dx2 = 2.0 / (bdx_u + fdx_u) * (
            (u[2:, :, :] - ui) / fdx_u - (ui - u[:-2, :, :]) / bdx_u)
        d2u_dy2 = 2.0 / (bdy_u + fdy_u) * (
            (u_py[:, 2:, :] - ui) / fdy_u - (ui - u_py[:, :-2, :]) / bdy_u)
        d2u_dz2 = 2.0 / (bdz_u + fdz_u) * (
            (u_pz[:, :, 2:] - ui) / fdz_u - (ui - u_pz[:, :, :-2]) / bdz_u)

        u_star[1:-1, :, :] = ui + dt * (
            -ui * du_dx - v_at_u * du_dy - w_at_u * du_dz
            + nu * (d2u_dx2 + d2u_dy2 + d2u_dz2))

        # ---- v[:, 1:-1, :]: interior y-faces  (nx, ny-1, nz) ----
        vi = v[:, 1:-1, :]

        u_at_v = 0.25 * (u[:-1, :-1, :] + u[:-1, 1:, :] +
                         u[1:,  :-1, :] + u[1:,  1:, :])
        w_at_v = 0.25 * (w[:, :-1, :-1] + w[:, :-1, 1:] +
                         w[:, 1:,  :-1] + w[:, 1:,  1:])

        dv_dy = np.where(vi > 0,
                         (v[:, 1:-1, :] - v[:, :-2, :]) / bdy_v,
                         (v[:, 2:,   :] - v[:, 1:-1, :]) / fdy_v)

        v_px = _pad(vi, var='v', axis=0, allow_lid=False)
        dv_dx = np.where(u_at_v > 0,
                         (vi             - v_px[:-2, :, :]) / bdx_v,
                         (v_px[2:, :, :] - vi             ) / fdx_v)

        v_pz = _pad(vi, var='v', axis=2, allow_lid=False)
        dv_dz = np.where(w_at_v > 0,
                         (vi             - v_pz[:, :, :-2]) / bdz_v,
                         (v_pz[:, :, 2:] - vi             ) / fdz_v)

        d2v_dy2 = 2.0 / (bdy_v + fdy_v) * (
            (v[:, 2:, :] - vi) / fdy_v - (vi - v[:, :-2, :]) / bdy_v)
        d2v_dx2 = 2.0 / (bdx_v + fdx_v) * (
            (v_px[2:, :, :] - vi) / fdx_v - (vi - v_px[:-2, :, :]) / bdx_v)
        d2v_dz2 = 2.0 / (bdz_v + fdz_v) * (
            (v_pz[:, :, 2:] - vi) / fdz_v - (vi - v_pz[:, :, :-2]) / bdz_v)

        v_star[:, 1:-1, :] = vi + dt * (
            -u_at_v * dv_dx - vi * dv_dy - w_at_v * dv_dz
            + nu * (d2v_dx2 + d2v_dy2 + d2v_dz2))

        # ---- w[:, :, 1:-1]: interior z-faces  (nx, ny, nz-1) ----
        wi = w[:, :, 1:-1]

        u_at_w = 0.25 * (u[:-1, :, :-1] + u[:-1, :, 1:] +
                         u[1:,  :, :-1] + u[1:,  :, 1:])
        v_at_w = 0.25 * (v[:, :-1, :-1] + v[:, :-1, 1:] +
                         v[:, 1:,  :-1] + v[:, 1:,  1:])

        dw_dz = np.where(wi > 0,
                         (w[:, :, 1:-1] - w[:, :, :-2]) / bdz_w,
                         (w[:, :, 2:]   - w[:, :, 1:-1]) / fdz_w)

        w_px = _pad(wi, var='w', axis=0, allow_lid=False)
        dw_dx = np.where(u_at_w > 0,
                         (wi             - w_px[:-2, :, :]) / bdx_w,
                         (w_px[2:, :, :] - wi             ) / fdx_w)

        w_py = _pad(wi, var='w', axis=1, allow_lid=False)
        dw_dy = np.where(v_at_w > 0,
                         (wi             - w_py[:, :-2, :]) / bdy_w,
                         (w_py[:, 2:, :] - wi             ) / fdy_w)

        d2w_dz2 = 2.0 / (bdz_w + fdz_w) * (
            (w[:, :, 2:] - wi) / fdz_w - (wi - w[:, :, :-2]) / bdz_w)
        d2w_dx2 = 2.0 / (bdx_w + fdx_w) * (
            (w_px[2:, :, :] - wi) / fdx_w - (wi - w_px[:-2, :, :]) / bdx_w)
        d2w_dy2 = 2.0 / (bdy_w + fdy_w) * (
            (w_py[:, 2:, :] - wi) / fdy_w - (wi - w_py[:, :-2, :]) / bdy_w)

        w_star[:, :, 1:-1] = wi + dt * (
            -u_at_w * dw_dx - v_at_w * dw_dy - wi * dw_dz
            + nu * (d2w_dx2 + d2w_dy2 + d2w_dz2))

        return u_star, v_star, w_star

    # ------------------------------------------------------------------ #
    # Boundary conditions
    # ------------------------------------------------------------------ #

    def _apply_bc(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> None:
        """
        Enforce velocity BCs on faces that sit ON walls:
          u-faces at i=0 / i=nx           (left / right walls)
          v-faces at j=0 / j=ny           (bottom / top walls)
          w-faces at k=0 / k=nz           (front / back walls)
        All other u/v/w values are real interior fluid; stencil ghosts for
        diffusion/advection are built inline in _advect_diffuse.
        """
        bct = self.domain.bc_type
        bcv = self.domain.bc_values
        u_in = float(bcv.get('inlet_u', 0.0))

        # Left face (x=0)
        if bct.get('left', 'no_slip') == 'inlet':
            u[0, :, :] = u_in
        else:
            u[0, :, :] = 0.0

        # Right face (x=nx)
        if bct.get('right', 'no_slip') == 'outlet':
            u[-1, :, :] = u[-2, :, :]
        else:
            u[-1, :, :] = 0.0

        # Bottom face (y=0): v is ON the wall (no normal flow); lid only
        # moves in x, so v[:, 0, :] is still zero on a lid wall.
        v[:, 0, :] = 0.0

        # Top face (y=ny)
        bt_top = bct.get('top', 'no_slip')
        if bt_top in ('outlet', 'outlet_v'):
            v[:, -1, :] = v[:, -2, :]
        else:
            v[:, -1, :] = 0.0

        # Front face (z=0) and back face (z=nz): w on the wall
        w[:, :,  0] = 0.0
        w[:, :, -1] = 0.0

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
        rho, dt = self.material.rho, self.dt

        div = ((u_star[1:, :, :] - u_star[:-1, :, :]) * self._inv_dx_cell
             + (v_star[:, 1:, :] - v_star[:, :-1, :]) * self._inv_dy_cell
             + (w_star[:, :, 1:] - w_star[:, :, :-1]) * self._inv_dz_cell)  # (nx,ny,nz)

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
        dt, rho = self.dt, self.material.rho
        α = self.relax

        u_new = u_star.copy()
        v_new = v_star.copy()
        w_new = w_star.copy()
        # Pressure gradient at an interior face uses the cell-centre-to-centre
        # distance ½·(dx[i-1] + dx[i]), not the cell width.
        u_new[1:-1, :, :] -= α * (dt / rho) * (p[1:,  :,  :] - p[:-1, :,  :]) / self._dx_pcorr
        v_new[:,  1:-1, :] -= α * (dt / rho) * (p[:,  1:,  :] - p[:,  :-1, :]) / self._dy_pcorr
        w_new[:,  :, 1:-1] -= α * (dt / rho) * (p[:,  :,  1:] - p[:,  :, :-1]) / self._dz_pcorr
        return u_new, v_new, w_new

    # ------------------------------------------------------------------ #
    # Build sparse 3-D Laplacian (once at init)
    # ------------------------------------------------------------------ #

    def _build_poisson(self):
        """
        Build 3-D Laplacian for ∇²p = rhs using Kronecker products.

        Column-major ordering: k = i + j*nx + kz*nx*ny.  All nx × ny × nz
        cells are real fluid; walls live on the bounding faces.  Each
        1-D Laplacian is a finite-volume discretisation that handles
        non-uniform cell widths:

            L[i, i-1] = 1 / (dx[i] · ½(dx[i-1] + dx[i]))
            L[i, i+1] = 1 / (dx[i] · ½(dx[i]   + dx[i+1]))
            L[i, i]   = −(above two)

        At i=0 / i=n−1 the missing-neighbour term is dropped — that is the
        Neumann ∂p/∂n = 0 wall condition.  On a uniform grid every entry
        reduces to ±1/h² and matches the previous scalar formulation.

        Outlet / solid cells get identity rows (Dirichlet p=0);
        a closed cavity pins one free cell to p=0.
        """
        nx, ny, nz = self.domain.nx, self.domain.ny, self.domain.nz
        dx_arr = self.domain.dx_arr
        dy_arr = self.domain.dy_arr
        dz_arr = self.domain.dz_arr
        bct = self.domain.bc_type
        N   = nx * ny * nz

        outlet_right  = bct.get('right')  == 'outlet'
        outlet_left   = bct.get('left')   == 'outlet'
        outlet_top    = bct.get('top')    in ('outlet', 'outlet_v')
        outlet_bottom = bct.get('bottom') in ('outlet', 'outlet_v')
        has_outlet    = outlet_right or outlet_left or outlet_top or outlet_bottom

        # ---- 1-D Laplacians with Neumann BC, non-uniform spacing aware ----
        def _nonuniform_lap_1d(spacing):
            n = spacing.size
            ax_minus = np.zeros(n)
            ax_plus  = np.zeros(n)
            if n >= 2:
                h_face       = 0.5 * (spacing[:-1] + spacing[1:])
                ax_minus[1:] = 1.0 / (spacing[1:]  * h_face)
                ax_plus[:-1] = 1.0 / (spacing[:-1] * h_face)
            diag = -(ax_minus + ax_plus)
            return spdiags([ax_minus[1:], diag, ax_plus[:-1]],
                           [-1, 0, 1], shape=(n, n), format='csr')

        Tx = _nonuniform_lap_1d(dx_arr)
        Ty = _nonuniform_lap_1d(dy_arr)
        Tz = _nonuniform_lap_1d(dz_arr)
        Ix = speye(nx, format='csr')
        Iy = speye(ny, format='csr')
        Iz = speye(nz, format='csr')

        # Full 3-D Laplacian (column-major: i fastest, then j, then kz)
        L = (kron(kron(Iz, Iy), Tx, format='csr') +
             kron(kron(Iz, Ty), Ix, format='csr') +
             kron(kron(Tz, Iy), Ix, format='csr'))
        L = L.tolil()

        rhs = np.zeros(N)

        Ia, Ja, Ka = np.mgrid[0:nx, 0:ny, 0:nz]
        Ia_f = Ia.ravel(); Ja_f = Ja.ravel(); Ka_f = Ka.ravel()
        flat = (Ia_f + Ja_f * nx + Ka_f * nx * ny).astype(np.intp)

        dirichlet_mask = np.zeros(N, dtype=bool)

        def _set_dirichlet(idx_arr):
            for k in idx_arr.tolist():
                L.rows[k] = [k]
                L.data[k] = [1.0]
            dirichlet_mask[idx_arr] = True

        if outlet_right:  _set_dirichlet(flat[Ia_f == nx - 1])
        if outlet_left:   _set_dirichlet(flat[Ia_f == 0])
        if outlet_top:    _set_dirichlet(flat[Ja_f == ny - 1])
        if outlet_bottom: _set_dirichlet(flat[Ja_f == 0])

        solid_flat = self.domain.solid.ravel(order='F')
        solid_idx  = flat[solid_flat]
        if solid_idx.size > 0:
            _set_dirichlet(solid_idx)

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
