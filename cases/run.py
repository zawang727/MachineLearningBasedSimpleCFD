"""
Generic runner for .cfd input files.

A .cfd file bundles fluid parameters, axis stretching, and the ASCII cell
map in one place — easier to share, diff, and tweak than scattered Python
case files.  See `Domain.from_text` docstring (cfd/domain.py) for the
full format reference.

Usage:
    python cases/run.py cases/lid_driven_cavity.cfd
    python cases/run.py cases/channel_flow.cfd --duration 25 --tol 1e-6
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cfd import Domain, Solver, Material, plot_fields, plot_velocity


def run(
    input_path: str,
    duration:   float | None = None,
    tol:        float        = 1e-6,
    out_dir:    str          = "results",
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    os.makedirs(out_dir, exist_ok=True)

    domain = Domain.from_file(input_path)

    rho = float(domain.bc_values.get('rho', 1.0))
    nu  = float(domain.bc_values.get('nu',  0.01))
    material = Material(rho=rho, nu=nu)
    solver   = Solver(domain, material)

    tag       = os.path.splitext(os.path.basename(input_path))[0]
    dy_arr    = domain.dy_arr
    dx_arr    = domain.dx_arr
    print(f"--- {input_path} ---")
    print(f"  grid:    {domain.nx} x {domain.ny}   "
          f"Lx={domain.Lx:.3g}  Ly={domain.Ly:.3g}")
    print(f"  dx:      min={dx_arr.min():.4f}  max={dx_arr.max():.4f}  "
          f"ratio={dx_arr.max()/dx_arr.min():.2f}")
    print(f"  dy:      min={dy_arr.min():.4f}  max={dy_arr.max():.4f}  "
          f"ratio={dy_arr.max()/dy_arr.min():.2f}")
    print(f"  BCs:     {domain.bc_type}")
    print(f"  params:  rho={rho}  nu={nu}  bc_values={domain.bc_values}")
    print(f"  dt:      {solver.dt:.5g}  (auto-CFL)")

    # Heuristic default: ~25 advective times across the longer axis.
    if duration is None:
        U_ref    = max(abs(domain.bc_values.get('inlet_u', 0.0)),
                       abs(domain.bc_values.get('lid_u',   1.0)), 1e-6)
        duration = 25.0 * max(domain.Lx, domain.Ly) / U_ref
    print(f"  run for: {duration:.3g} s")

    state = solver.run(duration, tol=tol, print_every=500)

    plot_fields(state, title=tag,
                save_path=os.path.join(out_dir, f"{tag}_fields.png"))
    plot_velocity(state, title=tag,
                  save_path=os.path.join(out_dir, f"{tag}_velocity.png"))
    print(f"  plots:   {out_dir}/{tag}_*.png")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a .cfd input file through the 2D solver.")
    p.add_argument('input',         type=str, help="path to .cfd file")
    p.add_argument('--duration',    type=float, default=None,
                   help="simulation time in seconds (default: 25 * L / U)")
    p.add_argument('--tol',         type=float, default=1e-6,
                   help="convergence tolerance on |du|/dt (default 1e-6)")
    p.add_argument('--out-dir',     type=str, default="results")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.input, args.duration, args.tol, args.out_dir)
