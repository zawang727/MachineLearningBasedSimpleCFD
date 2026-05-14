"""
Flow Around a Square Block — obstacle in channel flow.

Geometry:
  Inlet (>) on left, outlet (<) on right, no-slip walls top/bottom.
  A solid square block sits in the channel.
  Flow separates around the block; recirculation wake forms downstream.

Interesting physics:
  Re < ~50   : steady symmetric wake
  Re ~ 100   : asymmetric wake, possible weak oscillations
  Re > ~200  : unsteady (vortex shedding) — projection method captures this
"""
from __future__ import annotations
import os
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cfd import Domain, Solver, Material, plot_fields, plot_velocity


def _make_ascii(nx: int, ny: int,
                block_w: int, block_h: int,
                block_x: int, block_y: int) -> str:
    """
    Build ASCII map for channel with a solid block obstacle.
    block_x, block_y: left-bottom corner of block in interior cell coordinates.
    """
    # Fill interior with spaces
    grid = [[' '] * nx for _ in range(ny)]

    # Place block
    for i in range(block_x, min(block_x + block_w, nx)):
        for j in range(block_y, min(block_y + block_h, ny)):
            grid[j][i] = '*'

    top_bot = '#' * (nx + 2)
    lines   = [top_bot]
    for j in range(ny - 1, -1, -1):    # top to bottom in ASCII
        row = '>' + ''.join(grid[j]) + '<'
        lines.append(row)
    lines.append(top_bot)
    return '\n'.join(lines)


def run(
    Re:       float = 100.0,
    nx:       int   = 80,
    ny:       int   = 40,
    block_w:  int   = 4,
    block_h:  int   = 4,
    block_y_frac: float = 0.5,    # vertical centre of block as fraction of channel height
    U_in:     float = 1.0,
    duration: float | None = None,
    out_dir:  str   = "results",
    quiet:    bool  = False,
) -> object:
    """
    Run flow around a square block.
    Returns FlowState.
    """
    os.makedirs(out_dir, exist_ok=True)

    H   = 1.0
    dy  = H / ny
    dx  = dy
    nu  = U_in * H / Re

    # Block centred at x=1/4 of channel, y = block_y_frac
    block_x = max(0, int(0.25 * nx) - block_w // 2)
    block_y = max(0, int(block_y_frac * ny) - block_h // 2)

    ascii_map = _make_ascii(nx, ny, block_w, block_h, block_x, block_y)
    params    = {'inlet_u': U_in, 'rho': 1.0, 'nu': nu}
    domain    = Domain.from_ascii(ascii_map, params, dx=dx, dy=dy)
    material  = Material(rho=1.0, nu=nu)
    solver    = Solver(domain, material)

    if not quiet:
        print(f"Flow around block  Re={Re:.0f}  grid={nx}x{ny}  "
              f"block={block_w}x{block_h} at ({block_x},{block_y})  dt={solver.dt:.5f}")

    if duration is None:
        duration = 10.0 * nx * dx / U_in

    state = solver.run(duration, tol=1e-6, print_every=500 if not quiet else 100000)

    tag = f"Re{int(Re)}_by{int(block_y_frac*100)}"
    plot_fields(state,
                title=f"Flow around block  Re={Re:.0f}",
                save_path=os.path.join(out_dir, f"block_{tag}_fields.png"))
    plot_velocity(state,
                  title=f"Flow around block  Re={Re:.0f}  — streamlines",
                  save_path=os.path.join(out_dir, f"block_{tag}_streamlines.png"))

    if not quiet:
        print(f"  Saved plots to {out_dir}/block_{tag}_*.png")

    return state


def run_all_Re(nx: int = 80, ny: int = 40, out_dir: str = "results") -> dict:
    results = {}
    for Re in [50, 100, 200]:
        for by in [0.5]:
            print(f"\n=== Flow around block  Re={Re}  block_y={by} ===")
            results[(Re, by)] = run(Re=Re, nx=nx, ny=ny,
                                    block_y_frac=by, out_dir=out_dir)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--Re",           type=float, default=100)
    p.add_argument("--nx",           type=int,   default=80)
    p.add_argument("--ny",           type=int,   default=40)
    p.add_argument("--block-w",      type=int,   default=4)
    p.add_argument("--block-h",      type=int,   default=4)
    p.add_argument("--block-y-frac", type=float, default=0.5)
    p.add_argument("--duration",     type=float, default=None)
    p.add_argument("--out-dir",      type=str,   default="results")
    p.add_argument("--all-Re",       action="store_true")
    args = p.parse_args()

    if args.all_Re:
        run_all_Re(args.nx, args.ny, args.out_dir)
    else:
        run(args.Re, args.nx, args.ny, args.block_w, args.block_h,
            args.block_y_frac, out_dir=args.out_dir)
