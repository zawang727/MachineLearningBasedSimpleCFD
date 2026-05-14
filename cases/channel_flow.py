"""
Channel Flow (Poiseuille) — fully developed laminar pipe flow.

Geometry:
  Inlet (>) on left, outlet (<) on right.
  Top and bottom walls are no-slip.

Analytical steady-state profile:
  u(y) = 6 · U_avg · y · (H - y) / H²

where H is the channel height and U_avg is the mean inlet velocity.

Validation: compare numerical u(y) at outlet against analytical.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cfd import Domain, Solver, Material, plot_fields, plot_velocity


def _ascii_map(nx: int, ny: int) -> str:
    top_bot = "#" * (nx + 2)
    fluid   = ">" + " " * nx + "<"
    lines   = [top_bot] + [fluid] * ny + [top_bot]
    return "\n".join(lines)


def run(
    Re:       float = 100.0,
    nx:       int   = 80,
    ny:       int   = 20,
    U_avg:    float = 1.0,
    duration: float | None = None,
    out_dir:  str   = "results",
    quiet:    bool  = False,
) -> tuple:
    """
    Run Poiseuille channel flow.
    Returns (FlowState, max_error_fraction).
    """
    os.makedirs(out_dir, exist_ok=True)

    H   = 1.0              # channel height
    dy  = H / ny
    dx  = dy               # square cells
    nu  = U_avg * H / Re

    params = {'inlet_u': U_avg, 'rho': 1.0, 'nu': nu}
    ascii_map = _ascii_map(nx, ny)
    domain   = Domain.from_ascii(ascii_map, params, dx=dx, dy=dy)
    material = Material(rho=1.0, nu=nu)
    solver   = Solver(domain, material)

    if not quiet:
        print(f"Channel flow  Re={Re:.0f}  grid={nx}x{ny}  dt={solver.dt:.5f}")

    if duration is None:
        duration = 5.0 * nx * dx / U_avg   # ~5 flow-through times

    state = solver.run(duration, tol=1e-7, print_every=500 if not quiet else 100000)

    # --- Validation vs Poiseuille at outlet ---
    y_cell = (np.arange(ny) + 0.5) * dy
    u_num  = state.u_cell[-1, :]          # outlet profile
    u_ana  = 6.0 * U_avg * y_cell * (H - y_cell) / H ** 2
    max_err = np.max(np.abs(u_num - u_ana)) / (u_ana.max() + 1e-12)
    if not quiet:
        print(f"  Max error vs Poiseuille: {max_err*100:.2f}%")

    tag = f"Re{int(Re)}"
    plot_fields(state,
                title=f"Channel flow  Re={Re:.0f}",
                save_path=os.path.join(out_dir, f"channel_{tag}_fields.png"))
    plot_velocity(state,
                  title=f"Channel flow  Re={Re:.0f}  — streamlines",
                  save_path=os.path.join(out_dir, f"channel_{tag}_streamlines.png"))

    # Profile comparison plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(u_num, y_cell, 'b-o', ms=3, label='Numerical')
    ax.plot(u_ana, y_cell, 'r--',        label='Analytical (Poiseuille)')
    ax.set_xlabel('u [m/s]'); ax.set_ylabel('y [m]')
    ax.set_title(f'Channel flow Re={Re:.0f} — outlet u(y)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"channel_{tag}_profile.png"), dpi=150)
    plt.close()

    if not quiet:
        print(f"  Saved plots to {out_dir}/channel_{tag}_*.png")

    return state, max_err


def run_all_Re(nx: int = 80, ny: int = 20, out_dir: str = "results") -> dict:
    results = {}
    for Re in [50, 100, 200]:
        print(f"\n=== Channel flow  Re={Re} ===")
        state, err = run(Re=Re, nx=nx, ny=ny, out_dir=out_dir)
        results[Re] = (state, err)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--Re",       type=float, default=100)
    p.add_argument("--nx",       type=int,   default=80)
    p.add_argument("--ny",       type=int,   default=20)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--out-dir",  type=str,   default="results")
    p.add_argument("--all-Re",   action="store_true")
    args = p.parse_args()

    if args.all_Re:
        run_all_Re(args.nx, args.ny, args.out_dir)
    else:
        run(args.Re, args.nx, args.ny, out_dir=args.out_dir)
