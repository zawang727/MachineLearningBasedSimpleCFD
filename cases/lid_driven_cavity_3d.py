"""
3-D Lid-Driven Cavity.

ASCII extrude format (square cross-section extruded in z):

    ---------
    #       #
    #       #
    ---------   ← top wall moves at +x (lid)

All other walls (bottom, left, right, front, back) are no-slip.

Validation: compare xy-midplane u(y) centreline profile with
2-D benchmark (Ghia 1982) at Re=100, 400, 1000.

Run:
    python cases/lid_driven_cavity_3d.py --Re 100 --nx 32
    python cases/lid_driven_cavity_3d.py --Re 400 --nx 32 --nz 32
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cfd import Domain3D, Solver3D, Material, plot_fields_3d, plot_velocity_3d


# Ghia (1982) u-velocity along vertical centreline  Re=100, 400, 1000
_GHIA_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
           0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
           0.9688, 0.9766, 1.0000]
_GHIA_U = {
    100:  [-0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
           -0.15662, -0.21090, -0.20581, -0.13641,  0.00332,  0.23111,
            0.68717,  0.73722,  0.78871,  0.84123,  1.00000],
    400:  [-0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
           -0.32726, -0.17119, -0.11477,  0.02135,  0.16256,  0.29093,
            0.55892,  0.61756,  0.68439,  0.75837,  1.00000],
    1000: [-0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
           -0.27805, -0.10648, -0.06080,  0.05702,  0.18719,  0.33304,
            0.46604,  0.51117,  0.57492,  0.65928,  1.00000],
}

_ASCII_MAP = """\
----------
-        -
#        #
#        #
#        #
-        -
----------
"""


def run(
    Re:       float = 100.0,
    nx:       int   = 32,
    ny:       int | None = None,
    nz:       int | None = None,
    U_lid:    float = 1.0,
    duration: float | None = None,
    out_dir:  str   = "results",
    quiet:    bool  = False,
) -> 'FlowState3D':
    """
    Run 3-D lid-driven cavity.  Returns FlowState3D.
    Default grid: nx×nx×nx (cubic).
    """
    os.makedirs(out_dir, exist_ok=True)
    ny  = ny  or nx
    nz  = nz  or nx
    nu  = U_lid / Re
    dx  = 1.0 / nx
    dy  = 1.0 / ny
    dz  = 1.0 / nz

    # Use Domain3D.closed factory (cleaner than ASCII for a plain cavity)
    domain   = Domain3D.closed(nx, ny, nz, dx=dx, dy=dy, dz=dz,
                               params={'lid_u': U_lid, 'rho': 1.0, 'nu': nu})
    material = Material(rho=1.0, nu=nu)
    solver   = Solver3D(domain, material)

    if not quiet:
        N = nx * ny * nz
        print(f"3D Lid-driven cavity  Re={Re:.0f}  grid={nx}x{ny}x{nz}"
              f"  dt={solver.dt:.5f}  N={N}")
        if not solver._use_direct:
            print("  (iterative GMRES solver - large grid, may be slow)")

    if duration is None:
        duration = 2.0 / U_lid     # 2 flow-through times

    state = solver.run(duration, tol=1e-7,
                       print_every=200 if not quiet else 999999)

    tag = f"Re{int(Re)}_{nx}x{ny}x{nz}"
    plot_fields_3d(state,
                   title=f"3D Lid-driven cavity  Re={Re:.0f}",
                   save_path=os.path.join(out_dir, f"cavity3d_{tag}_fields.png"))
    plot_velocity_3d(state,
                     title=f"3D Lid-driven cavity  Re={Re:.0f}  — midplane speed",
                     save_path=os.path.join(out_dir, f"cavity3d_{tag}_speed.png"))

    # Centreline u(y) at xy-midplane, x=0.5
    kz_mid  = nz // 2
    ix_mid  = nx // 2
    u_cline = state.u_cell[ix_mid, :, kz_mid]      # (ny,)
    y_vals  = (np.arange(ny) + 0.5) * dy

    int_Re = int(Re)
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(u_cline / U_lid, y_vals, 'b-o', ms=2, label='3D CFD (midplane)')
    if int_Re in _GHIA_U:
        ax.plot(_GHIA_U[int_Re], _GHIA_Y, 'r--s', ms=4, label='Ghia (1982)')
    ax.set_xlabel('u / U_lid'); ax.set_ylabel('y')
    ax.set_title(f'3D Cavity Re={Re:.0f} — centreline u(y)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cavity3d_{tag}_cline.png"), dpi=150)
    plt.close()

    if not quiet:
        print(f"  Plots saved to {out_dir}/cavity3d_{tag}_*.png")

    return state


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--Re",       type=float, default=100)
    p.add_argument("--nx",       type=int,   default=32)
    p.add_argument("--ny",       type=int,   default=None)
    p.add_argument("--nz",       type=int,   default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--out-dir",  type=str,   default="results")
    args = p.parse_args()
    run(args.Re, args.nx, args.ny, args.nz,
        duration=args.duration, out_dir=args.out_dir)
