"""
Generic runner for .cfd input files (2-D and 3-D).

A .cfd file bundles fluid parameters, axis stretching, and the ASCII cell
map in one place — easier to share, diff, and tweak than scattered Python
case files.  See `Domain.from_text` and `Domain3D.from_text` docstrings
(cfd/domain.py, cfd/domain3d.py) for the full format reference.

Dispatch: a file with an `nz:` key in the header is parsed as a 3-D case;
otherwise it is 2-D.

Usage:
    python cases/run.py cases/lid_driven_cavity.cfd
    python cases/run.py cases/lid_driven_cavity_3d.cfd
    python cases/run.py cases/channel_flow.cfd --duration 25 --tol 1e-6
"""
from __future__ import annotations
import argparse, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cfd import (Domain, Domain3D, Solver, Solver3D, Material,
                 plot_fields, plot_velocity,
                 plot_fields_3d, plot_velocity_3d)


_NZ_HEADER = re.compile(r'^\s*nz\s*:', re.MULTILINE)


def _is_3d(input_path: str) -> bool:
    """A file with an `nz:` line in the header (above '---') is 3-D."""
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    header = text.split('---', 1)[0] if '---' in text else text
    return bool(_NZ_HEADER.search(header))


def _run_2d(input_path: str, duration, tol, out_dir):
    domain   = Domain.from_file(input_path)
    rho      = float(domain.bc_values.get('rho', 1.0))
    nu       = float(domain.bc_values.get('nu',  0.01))
    material = Material(rho=rho, nu=nu)
    solver   = Solver(domain, material)

    tag = os.path.splitext(os.path.basename(input_path))[0]
    print(f"--- {input_path} (2D) ---")
    _print_axis('dx', domain.dx_arr)
    _print_axis('dy', domain.dy_arr)
    print(f"  grid:    {domain.nx} × {domain.ny}   "
          f"Lx={domain.Lx:.3g}  Ly={domain.Ly:.3g}")
    print(f"  BCs:     {domain.bc_type}")
    print(f"  params:  rho={rho}  nu={nu}  bc_values={domain.bc_values}")
    print(f"  dt:      {solver.dt:.5g}  (auto-CFL)")
    if duration is None:
        duration = _default_duration(domain.bc_values,
                                      max(domain.Lx, domain.Ly))
    print(f"  run for: {duration:.3g} s")

    state = solver.run(duration, tol=tol, print_every=500)
    plot_fields(state,   title=tag,
                save_path=os.path.join(out_dir, f"{tag}_fields.png"))
    plot_velocity(state, title=tag,
                  save_path=os.path.join(out_dir, f"{tag}_velocity.png"))
    print(f"  plots:   {out_dir}/{tag}_*.png")


def _run_3d(input_path: str, duration, tol, out_dir):
    domain   = Domain3D.from_file(input_path)
    rho      = float(domain.bc_values.get('rho', 1.0))
    nu       = float(domain.bc_values.get('nu',  0.01))
    material = Material(rho=rho, nu=nu)
    solver   = Solver3D(domain, material)

    tag = os.path.splitext(os.path.basename(input_path))[0]
    print(f"--- {input_path} (3D) ---")
    _print_axis('dx', domain.dx_arr)
    _print_axis('dy', domain.dy_arr)
    _print_axis('dz', domain.dz_arr)
    print(f"  grid:    {domain.nx} × {domain.ny} × {domain.nz}   "
          f"Lx={domain.Lx:.3g}  Ly={domain.Ly:.3g}  Lz={domain.Lz:.3g}")
    print(f"  BCs:     {domain.bc_type}")
    print(f"  params:  rho={rho}  nu={nu}  bc_values={domain.bc_values}")
    print(f"  dt:      {solver.dt:.5g}  (auto-CFL)")
    if duration is None:
        duration = _default_duration(domain.bc_values,
                                      max(domain.Lx, domain.Ly, domain.Lz))
    print(f"  run for: {duration:.3g} s")

    state = solver.run(duration, tol=tol, print_every=200)
    plot_fields_3d(state,   title=tag,
                   save_path=os.path.join(out_dir, f"{tag}_fields.png"))
    plot_velocity_3d(state, title=tag,
                     save_path=os.path.join(out_dir, f"{tag}_velocity.png"))
    print(f"  plots:   {out_dir}/{tag}_*.png")


def _print_axis(name, arr):
    ratio = arr.max() / arr.min()
    print(f"  {name}:      min={arr.min():.4f}  max={arr.max():.4f}  "
          f"ratio={ratio:.2f}")


def _default_duration(bc_values, L):
    U_ref = max(abs(bc_values.get('inlet_u', 0.0)),
                abs(bc_values.get('lid_u',   1.0)), 1e-6)
    return 25.0 * L / U_ref


def run(input_path: str, duration=None, tol=1e-6, out_dir="results"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    os.makedirs(out_dir, exist_ok=True)
    if _is_3d(input_path):
        _run_3d(input_path, duration, tol, out_dir)
    else:
        _run_2d(input_path, duration, tol, out_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a .cfd input file through the 2-D or 3-D solver.")
    p.add_argument('input',         type=str, help="path to .cfd file")
    p.add_argument('--duration',    type=float, default=None,
                   help="simulation time in seconds (default: 25 * L / U)")
    p.add_argument('--tol',         type=float, default=1e-6)
    p.add_argument('--out-dir',     type=str,   default="results")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.input, args.duration, args.tol, args.out_dir)
