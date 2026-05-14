"""
3-D Channel Flow (square-duct Poiseuille).

ASCII extrude format:

    ########
    >      <
    >      <
    ########

Extruded in z (square cross-section H×H).

Analytical laminar profile for a square duct uses an infinite series;
as a practical validation we compare the maximum velocity at the
centreline with the theoretical value for a circular pipe of radius R:
    u_max (circular) = 2 U_avg
For a square duct, u_max ≈ 2.096 U_avg (from tabulated data).
We report the normalised centreline velocity instead.

Run:
    python cases/channel_flow_3d.py --Re 100
    python cases/channel_flow_3d.py --Re 200 --nx 40 --ny 20 --nz 20
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cfd import Domain3D, Solver3D, Material, plot_fields_3d, plot_velocity_3d


def _ascii_map(nx: int, ny: int) -> str:
    top_bot = "#" * (nx + 2)
    fluid   = ">" + " " * nx + "<"
    return "\n".join([top_bot] + [fluid] * ny + [top_bot])


def run(
    Re:       float = 100.0,
    nx:       int   = 40,
    ny:       int   = 12,
    nz:       int   = 12,
    U_avg:    float = 1.0,
    duration: float | None = None,
    out_dir:  str   = "results",
    quiet:    bool  = False,
) -> tuple:
    """
    Run 3-D square-duct channel flow.
    Returns (FlowState3D, centreline_u_normalised).
    """
    os.makedirs(out_dir, exist_ok=True)

    H  = 1.0           # duct height = duct width
    dy = H / ny
    dx = dy
    dz = H / nz
    nu = U_avg * H / Re

    params    = {'inlet_u': U_avg, 'rho': 1.0, 'nu': nu}
    ascii_map = _ascii_map(nx, ny)
    domain    = Domain3D.from_ascii(ascii_map, params, dx=dx, dy=dy, dz=dz, nz=nz)
    material  = Material(rho=1.0, nu=nu)
    solver    = Solver3D(domain, material)

    if not quiet:
        N = nx * ny * nz
        print(f"3D Channel flow  Re={Re:.0f}  grid={nx}x{ny}x{nz}"
              f"  dt={solver.dt:.5f}  N={N}")

    if duration is None:
        duration = 5.0 * nx * dx / U_avg

    state = solver.run(duration, tol=1e-7,
                       print_every=200 if not quiet else 999999)

    # Centreline velocity at outlet (x=nx-1, y=ny//2, z=nz//2)
    u_cline = state.u_cell[-1, ny // 2, nz // 2]
    u_norm  = float(u_cline) / U_avg

    if not quiet:
        print(f"  Centreline u/U_avg at outlet = {u_norm:.4f}"
              f"  (square duct theory ~2.096)")

    tag = f"Re{int(Re)}_{nx}x{ny}x{nz}"
    plot_fields_3d(state,
                   title=f"3D Channel  Re={Re:.0f}",
                   save_path=os.path.join(out_dir, f"channel3d_{tag}_fields.png"))
    plot_velocity_3d(state,
                     title=f"3D Channel  Re={Re:.0f}",
                     save_path=os.path.join(out_dir, f"channel3d_{tag}_speed.png"))

    # Cross-section u-profile at outlet, z-midplane
    kz_mid   = nz // 2
    y_vals   = (np.arange(ny) + 0.5) * dy
    u_prof_y = state.u_cell[-1, :, kz_mid]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(u_prof_y / U_avg, y_vals, 'b-o', ms=2, label='CFD (z-midplane)')
    ax.set_xlabel('u / U_avg'); ax.set_ylabel('y')
    ax.set_title(f'3D Channel Re={Re:.0f} — outlet u(y) at z-midplane')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"channel3d_{tag}_profile.png"), dpi=150)
    plt.close()

    if not quiet:
        print(f"  Plots saved to {out_dir}/channel3d_{tag}_*.png")

    return state, u_norm


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--Re",       type=float, default=100)
    p.add_argument("--nx",       type=int,   default=40)
    p.add_argument("--ny",       type=int,   default=12)
    p.add_argument("--nz",       type=int,   default=12)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--out-dir",  type=str,   default="results")
    args = p.parse_args()
    run(args.Re, args.nx, args.ny, args.nz,
        duration=args.duration, out_dir=args.out_dir)
