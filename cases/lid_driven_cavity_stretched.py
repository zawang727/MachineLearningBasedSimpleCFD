"""
Lid-Driven Cavity on a tanh-stretched grid (Phase 1 mesh demo).

Demonstrates that the 2D solver works on a non-uniform structured grid
that clusters cells near the no-slip walls.  Same physics as
`lid_driven_cavity.py` — single primary vortex; compared to Ghia (1982).

Run:
    python cases/lid_driven_cavity_stretched.py --Re 400 --nx 32 --ny 32
    python cases/lid_driven_cavity_stretched.py --Re 100 --beta-y 3.0
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cfd import Domain, Solver, Material, plot_fields, plot_velocity


# Ghia (1982) u-velocity along vertical centreline.
_GHIA_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
           0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
           0.9688, 0.9766, 1.0000]
_GHIA_U = {
    100:  [ 0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
           -0.15662, -0.21090, -0.20581, -0.13641,  0.00332,  0.23111,
            0.68717,  0.73722,  0.78871,  0.84123,  1.00000],
    400:  [ 0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
           -0.32726, -0.17119, -0.11477,  0.02135,  0.16256,  0.29093,
            0.55892,  0.61756,  0.68439,  0.75837,  1.00000],
    1000: [ 0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
           -0.27805, -0.10648, -0.06080,  0.05702,  0.18719,  0.33304,
            0.46604,  0.51117,  0.57492,  0.65928,  1.00000],
}


def run(
    Re:       float = 400.0,
    nx:       int   = 32,
    ny:       int   = 32,
    beta_x:   float = 1.0,
    beta_y:   float = 2.5,
    U_lid:    float = 1.0,
    duration: float | None = None,
    out_dir:  str   = "results",
    quiet:    bool  = False,
):
    os.makedirs(out_dir, exist_ok=True)
    nu = U_lid / Re

    domain   = Domain.stretched_closed(nx, ny, beta_x=beta_x, beta_y=beta_y,
                                        params={'lid_u': U_lid, 'rho': 1.0, 'nu': nu})
    material = Material(rho=1.0, nu=nu)
    solver   = Solver(domain, material)

    if not quiet:
        dy_min = float(domain.dy_arr.min())
        dy_max = float(domain.dy_arr.max())
        print(f"Stretched cavity Re={Re:.0f}  grid={nx}x{ny}  "
              f"beta=(x={beta_x},y={beta_y})  "
              f"dy_min={dy_min:.4f} dy_max={dy_max:.4f}  "
              f"ratio={dy_max/dy_min:.2f}  dt={solver.dt:.5f}")

    if duration is None:
        duration = max(20.0, 60.0 / U_lid)

    state = solver.run(duration, tol=1e-6,
                       print_every=500 if not quiet else 100000)

    tag = f"Re{int(Re)}_{nx}x{ny}_betay{beta_y:g}"
    plot_fields(state,
                title=f"Stretched cavity  Re={Re:.0f}  βy={beta_y}",
                save_path=os.path.join(out_dir, f"cavity_stretched_{tag}_fields.png"))
    plot_velocity(state,
                  title=f"Stretched cavity  Re={Re:.0f}  βy={beta_y}",
                  save_path=os.path.join(out_dir, f"cavity_stretched_{tag}_streamlines.png"))

    # ---- Validation: vertical centreline u(y) vs Ghia ----
    ix_mid  = nx // 2
    u_cline = state.u_cell[ix_mid, :]
    y_vals  = domain.y_cell        # honest non-uniform y positions

    int_Re = int(Re)
    fig, ax = plt.subplots(figsize=(5.5, 6))
    ax.plot(u_cline / U_lid, y_vals, 'b-o', ms=3, label=f'Stretched ({nx}x{ny})')
    if int_Re in _GHIA_U:
        ax.plot(_GHIA_U[int_Re], _GHIA_Y, 'r--s', ms=4, label='Ghia (1982)')
    ax.set_xlabel('u / U_lid'); ax.set_ylabel('y')
    ax.set_title(f'Stretched cavity Re={Re:.0f} — centreline u(y)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    cline_path = os.path.join(out_dir, f"cavity_stretched_{tag}_cline.png")
    plt.savefig(cline_path, dpi=150)
    plt.close()

    if not quiet:
        print(f"  Plots saved to {out_dir}/cavity_stretched_{tag}_*.png")

    return state


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--Re",       type=float, default=400)
    p.add_argument("--nx",       type=int,   default=32)
    p.add_argument("--ny",       type=int,   default=32)
    p.add_argument("--beta-x",   type=float, default=1.0)
    p.add_argument("--beta-y",   type=float, default=2.5)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--out-dir",  type=str,   default="results")
    args = p.parse_args()
    run(args.Re, args.nx, args.ny, args.beta_x, args.beta_y,
        duration=args.duration, out_dir=args.out_dir)
