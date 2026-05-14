"""
Run real 3-D test cases and generate 3-D plots.

Cases
-----
1. Lid-driven cavity   Re=100  32x32x32  -- validation vs Ghia (1982)
2. Lid-driven cavity   Re=400  32x32x32  -- higher Re recirculation
3. Square-duct channel Re=100  40x20x20  -- Poiseuille, analytical check

Outputs (all in results/)
-----
  cavity3d_Re100_summary.png     -- 2-panel: speed slices + velocity vectors
  cavity3d_Re100_speed_slices.png
  cavity3d_Re100_pressure_slices.png
  cavity3d_Re100_vectors.png
  cavity3d_Re400_summary.png
  cavity3d_Re400_speed_slices.png
  cavity3d_Re400_vectors.png
  channel3d_Re100_summary.png
  channel3d_Re100_speed_slices.png
  channel3d_Re100_vectors.png
  (+ standard 2-D midplane plots from the individual case runners)

Run:
    python run_3d_testcases.py
    python run_3d_testcases.py --out-dir results --nx 24
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from cfd import (Domain3D, Solver3D, Material,
                 plot_3d_slices, plot_3d_vectors, plot_3d_case_summary,
                 plot_fields_3d)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _run_cavity(Re: float, nx: int, out_dir: str, duration: float = None):
    """Run lid-driven cavity and return FlowState3D."""
    from cases.lid_driven_cavity_3d import run
    print(f"\n{'='*60}")
    print(f"  3D Lid-driven cavity  Re={Re:.0f}  grid={nx}x{nx}x{nx}")
    print(f"{'='*60}")
    state = run(Re=Re, nx=nx, ny=nx, nz=nx,
                duration=duration, out_dir=out_dir, quiet=False)
    return state


def _run_channel(Re: float, nx: int, ny: int, nz: int, out_dir: str):
    """Run square-duct channel flow and return (FlowState3D, u_norm)."""
    from cases.channel_flow_3d import run
    print(f"\n{'='*60}")
    print(f"  3D Channel flow  Re={Re:.0f}  grid={nx}x{ny}x{nz}")
    print(f"{'='*60}")
    state, u_norm = run(Re=Re, nx=nx, ny=ny, nz=nz, out_dir=out_dir, quiet=False)
    return state, u_norm


def _save_all_plots(state, tag: str, title: str, out_dir: str) -> None:
    """Generate all 3-D plots for a given flow state."""

    plot_3d_case_summary(
        state,
        title=title,
        save_path=os.path.join(out_dir, f'{tag}_summary.png'))
    print(f"  Saved {tag}_summary.png")

    plot_3d_slices(
        state, field='speed', cmap_name='viridis',
        title=f'{title} -- speed',
        save_path=os.path.join(out_dir, f'{tag}_speed_slices.png'))
    print(f"  Saved {tag}_speed_slices.png")

    plot_3d_slices(
        state, field='pressure', cmap_name='RdBu_r', symmetric=True,
        title=f'{title} -- pressure',
        save_path=os.path.join(out_dir, f'{tag}_pressure_slices.png'))
    print(f"  Saved {tag}_pressure_slices.png")

    plot_3d_vectors(
        state,
        title=f'{title} -- velocity vectors',
        save_path=os.path.join(out_dir, f'{tag}_vectors.png'))
    print(f"  Saved {tag}_vectors.png")

    plot_fields_3d(
        state,
        title=title,
        save_path=os.path.join(out_dir, f'{tag}_fields.png'))
    print(f"  Saved {tag}_fields.png")


# ------------------------------------------------------------------ #
# Centreline validation plot (cavity vs Ghia 1982)
# ------------------------------------------------------------------ #

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
}


def _plot_centreline_validation(state, Re: float, U_lid: float,
                                 tag: str, out_dir: str) -> None:
    dom   = state.domain
    nx, ny, nz = dom.nx, dom.ny, dom.nz
    kz_mid = nz // 2
    ix_mid = nx // 2
    y_vals = (np.arange(ny) + 0.5) * dom.dy
    u_cline = state.u_cell[ix_mid, :, kz_mid]

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(u_cline / U_lid, y_vals, 'b-o', ms=3, lw=1.5, label='3D CFD (z-midplane)')
    int_Re = int(Re)
    if int_Re in _GHIA_U:
        ax.plot(_GHIA_U[int_Re], _GHIA_Y, 'r--s', ms=5, lw=1.5, label='Ghia (1982) 2D')
    ax.set_xlabel('u / U_lid')
    ax.set_ylabel('y')
    ax.set_title(f'Lid cavity Re={Re:.0f} -- centreline u(y)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{tag}_cline_validation.png'), dpi=150)
    plt.close()
    print(f"  Saved {tag}_cline_validation.png")


def _plot_channel_validation(state, Re: float, U_avg: float,
                              tag: str, out_dir: str) -> None:
    """Compare outlet u(y) profile vs analytical 2-D Poiseuille at z-midplane."""
    dom = state.domain
    nx, ny, nz = dom.nx, dom.ny, dom.nz
    dy = dom.dy
    H  = ny * dy
    kz_mid = nz // 2
    y_vals = (np.arange(ny) + 0.5) * dy
    u_cfd  = state.u_cell[-1, :, kz_mid]
    # 2-D Poiseuille (circular/infinite slot) upper bound
    u_ana  = 6.0 * U_avg * y_vals * (H - y_vals) / H**2

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(u_cfd / U_avg, y_vals, 'b-o', ms=3, lw=1.5, label='3D CFD outlet (z-mid)')
    ax.plot(u_ana / U_avg, y_vals, 'r--',       lw=1.5, label='2D Poiseuille (reference)')
    ax.set_xlabel('u / U_avg')
    ax.set_ylabel('y')
    ax.set_title(f'Channel Re={Re:.0f} -- outlet u(y) at z-midplane')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{tag}_profile_validation.png'), dpi=150)
    plt.close()
    print(f"  Saved {tag}_profile_validation.png")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(nx_cav: int = 32, out_dir: str = 'results') -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ---- Case 1: Lid-driven cavity Re=100 ----
    state_c100 = _run_cavity(Re=100, nx=nx_cav, out_dir=out_dir, duration=10.0)
    tag = f'cavity3d_Re100_{nx_cav}x{nx_cav}x{nx_cav}'
    _save_all_plots(state_c100, tag,
                    f'3D Lid-driven cavity  Re=100  {nx_cav}^3', out_dir)
    _plot_centreline_validation(state_c100, Re=100, U_lid=1.0, tag=tag, out_dir=out_dir)

    # ---- Case 2: Lid-driven cavity Re=400 ----
    state_c400 = _run_cavity(Re=400, nx=nx_cav, out_dir=out_dir, duration=10.0)
    tag = f'cavity3d_Re400_{nx_cav}x{nx_cav}x{nx_cav}'
    _save_all_plots(state_c400, tag,
                    f'3D Lid-driven cavity  Re=400  {nx_cav}^3', out_dir)
    _plot_centreline_validation(state_c400, Re=400, U_lid=1.0, tag=tag, out_dir=out_dir)

    # ---- Case 3: Channel flow Re=100 ----
    nx_ch, ny_ch, nz_ch = max(40, nx_cav), max(16, nx_cav // 2), max(16, nx_cav // 2)
    state_ch, u_norm = _run_channel(Re=100, nx=nx_ch, ny=ny_ch, nz=nz_ch, out_dir=out_dir)
    print(f"  Centreline u/U_avg = {u_norm:.4f}  (square duct theory ~2.096)")
    tag = f'channel3d_Re100_{nx_ch}x{ny_ch}x{nz_ch}'
    _save_all_plots(state_ch, tag,
                    f'3D Channel flow  Re=100  {nx_ch}x{ny_ch}x{nz_ch}', out_dir)
    _plot_channel_validation(state_ch, Re=100, U_avg=1.0, tag=tag, out_dir=out_dir)

    print(f"\nAll plots saved to {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=str, default='results')
    p.add_argument('--nx',     type=int, default=32,
                   help='Grid size for cavity (nx x nx x nx)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(nx_cav=args.nx, out_dir=args.out_dir)
