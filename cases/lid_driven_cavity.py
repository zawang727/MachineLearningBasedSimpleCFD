"""
Lid-Driven Cavity — classic incompressible flow benchmark.

Geometry (nx × ny closed square):
  Top wall moves at U_lid in +x direction.
  All other walls are no-slip (u=v=0).
  No inlet or outlet — closed cavity.

Validation: Re = U_lid·L/ν.
  Re=100   → single primary vortex, steady.
  Re=400   → primary vortex + two corner eddies.
  Re=1000  → larger secondary eddies visible.

Reference: Ghia, Ghia, Shin (1982) — centre-line velocity profiles.
"""
from __future__ import annotations
import os
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cfd import Domain, Solver, Material, plot_fields, plot_velocity


_ASCII_MAP = """\
-{fluid}-
#{fluid}#
#{fluid}#
#{fluid}#
-{fluid}-
"""


def run(
    Re:      float = 100.0,
    nx:      int   = 64,
    ny:      int   = 64,
    duration: float | None = None,
    out_dir:  str   = "results",
    quiet:    bool  = False,
) -> object:
    """
    Run lid-driven cavity at given Reynolds number.
    Returns FlowState (steady-state or after `duration` time units).
    """
    os.makedirs(out_dir, exist_ok=True)

    L   = 1.0                     # cavity side length
    dx  = L / nx
    U   = 1.0                     # lid velocity
    nu  = U * L / Re

    domain   = Domain.closed(nx, ny, dx=dx,
                              params={'lid_u': U, 'rho': 1.0, 'nu': nu})
    material = Material(rho=1.0, nu=nu)
    solver   = Solver(domain, material)

    if not quiet:
        print(f"Lid-driven cavity  Re={Re:.0f}  grid={nx}x{ny}  dt={solver.dt:.5f}")

    if duration is None:
        duration = max(20.0, 50.0 * L / U)

    state = solver.run(duration, tol=1e-6, print_every=500 if not quiet else 100000)

    tag = f"Re{int(Re)}"
    plot_fields(state,
                title=f"Lid-driven cavity  Re={Re:.0f}",
                save_path=os.path.join(out_dir, f"cavity_{tag}_fields.png"))
    plot_velocity(state,
                  title=f"Lid-driven cavity  Re={Re:.0f}  — streamlines",
                  save_path=os.path.join(out_dir, f"cavity_{tag}_streamlines.png"))

    if not quiet:
        print(f"  Saved plots to {out_dir}/cavity_{tag}_*.png")

    return state


def run_all_Re(nx: int = 64, ny: int = 64, out_dir: str = "results") -> dict:
    """Run Re ∈ {100, 400, 1000} and return dict of FlowStates."""
    results = {}
    for Re in [100, 400, 1000]:
        print(f"\n=== Lid-driven cavity  Re={Re} ===")
        results[Re] = run(Re=Re, nx=nx, ny=ny, out_dir=out_dir)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--Re",       type=float, default=400)
    p.add_argument("--nx",       type=int,   default=64)
    p.add_argument("--ny",       type=int,   default=64)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--out-dir",  type=str,   default="results")
    p.add_argument("--all-Re",   action="store_true")
    args = p.parse_args()

    if args.all_Re:
        run_all_Re(args.nx, args.ny, args.out_dir)
    else:
        run(args.Re, args.nx, args.ny, args.duration, args.out_dir)
