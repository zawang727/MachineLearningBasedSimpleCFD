from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _mid(arr3d: np.ndarray, axis: int) -> np.ndarray:
    """Return the middle slice along `axis`."""
    idx = arr3d.shape[axis] // 2
    return np.take(arr3d, idx, axis=axis)


def plot_velocity_3d(state, title: str = "", save_path: str = "") -> None:
    """
    Two-panel midplane speed maps.
    Left : xy-midplane  (kz = nz//2)
    Right: xz-midplane  (j  = ny//2)
    """
    dom = state.domain
    x = (np.arange(dom.nx) + 0.5) * dom.dx
    y = (np.arange(dom.ny) + 0.5) * dom.dy
    z = (np.arange(dom.nz) + 0.5) * dom.dz

    speed = state.speed  # (nx, ny, nz)
    spd_xy = _mid(speed, axis=2).T    # (ny, nx) for contourf(x, y, ...)
    spd_xz = _mid(speed, axis=1).T    # (nz, nx)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, xlabel, ylabel, extent_x, extent_y, panel in [
        (axes[0], spd_xy, 'x', 'y', x, y, f'xy-plane (z={z[dom.nz//2]:.2f})'),
        (axes[1], spd_xz, 'x', 'z', x, z, f'xz-plane (y={y[dom.ny//2]:.2f})'),
    ]:
        cf = ax.contourf(extent_x, extent_y, data, levels=20, cmap='viridis')
        fig.colorbar(cf, ax=ax, label='Speed [m/s]')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(panel)
        ax.set_aspect('equal')

    if title:
        fig.suptitle(title, fontsize=12)
    try:
        plt.tight_layout()
    except Exception:
        pass
    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        except Exception:
            plt.savefig(save_path, dpi=150)
    plt.close()


def plot_fields_3d(state, title: str = "", save_path: str = "") -> None:
    """
    3×2 panel: rows = speed / pressure / w_cell
               cols = xy-midplane / xz-midplane
    """
    dom = state.domain
    x = (np.arange(dom.nx) + 0.5) * dom.dx
    y = (np.arange(dom.ny) + 0.5) * dom.dy
    z = (np.arange(dom.nz) + 0.5) * dom.dz

    fields = [
        (state.speed,  'Speed',    'viridis',  'Speed [m/s]',    False),
        (state.p,      'Pressure', 'RdBu_r',   'Pressure [Pa]',  True),
        (state.w_cell, 'w-vel',    'bwr',       'w [m/s]',        True),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    col_labels = [f'xy-plane (z={z[dom.nz//2]:.2f})',
                  f'xz-plane (y={y[dom.ny//2]:.2f})']

    for r, (field, name, cmap, clabel, symmetric) in enumerate(fields):
        slices = [_mid(field, axis=2).T,   # (ny, nx)
                  _mid(field, axis=1).T]   # (nz, nx)
        extents_x = [x, x]
        extents_y = [y, z]

        vmax = max(np.abs(field).max(), 1e-12)
        vmin = -vmax if symmetric else 0.0

        for c, (data, ex, ey, col_lbl) in enumerate(
                zip(slices, extents_x, extents_y, col_labels)):
            ax = axes[r, c]
            cf = ax.contourf(ex, ey, data, levels=20, cmap=cmap,
                             vmin=vmin, vmax=vmax)
            fig.colorbar(cf, ax=ax, label=clabel, fraction=0.046)
            if r == 0:
                ax.set_title(col_lbl, fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=10)
            ax.set_xlabel('x')
            ax.set_aspect('equal')

    if title:
        fig.suptitle(title, fontsize=12)
    try:
        plt.tight_layout()
    except Exception:
        pass
    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        except Exception:
            plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison_3d(
    state_nn,
    state_cfd,
    title: str = "",
    save_path: str = "",
) -> None:
    """
    4×2 comparison: rows = u/v/w/p, cols = NN / CFD (xy midplane).
    """
    dom  = state_cfd.domain
    x    = (np.arange(dom.nx) + 0.5) * dom.dx
    y    = (np.arange(dom.ny) + 0.5) * dom.dy

    rows = [
        ('u-velocity', state_nn.u_cell, state_cfd.u_cell, 'RdBu_r'),
        ('v-velocity', state_nn.v_cell, state_cfd.v_cell, 'RdBu_r'),
        ('w-velocity', state_nn.w_cell, state_cfd.w_cell, 'RdBu_r'),
        ('pressure',   state_nn.p,      state_cfd.p,      'RdBu_r'),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 17))
    col_labels = ['NN prediction', 'CFD solver']

    for r, (name, nn_f, cfd_f, cmap) in enumerate(rows):
        vmax = max(np.abs(cfd_f).max(), 1e-12)
        vmin = -vmax
        for c, (field, col_lbl) in enumerate(zip([nn_f, cfd_f], col_labels)):
            data = _mid(field, axis=2).T  # xy-midplane
            ax   = axes[r, c]
            cf   = ax.contourf(x, y, data, levels=20, cmap=cmap,
                               vmin=vmin, vmax=vmax)
            fig.colorbar(cf, ax=ax, fraction=0.046)
            if r == 0:
                ax.set_title(col_lbl, fontsize=11)
            if c == 0:
                ax.set_ylabel(name, fontsize=10)
            ax.set_aspect('equal')

    mae_u = float(np.mean(np.abs(state_nn.u_cell - state_cfd.u_cell)))
    mae_v = float(np.mean(np.abs(state_nn.v_cell - state_cfd.v_cell)))
    mae_w = float(np.mean(np.abs(state_nn.w_cell - state_cfd.w_cell)))
    mae_p = float(np.mean(np.abs(state_nn.p      - state_cfd.p)))
    subtitle = f'MAE  u={mae_u:.4f}  v={mae_v:.4f}  w={mae_w:.4f}  p={mae_p:.4f}'
    full_title = f'{title}\n{subtitle}' if title else subtitle
    fig.suptitle(full_title, fontsize=10)
    try:
        plt.tight_layout()
    except Exception:
        pass
    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        except Exception:
            plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history_3d(train_losses: list, val_losses: list,
                              save_path: str = "") -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train MAE')
    if val_losses:
        plt.plot(val_losses, label='Val MAE')
    plt.xlabel('Epoch'); plt.ylabel('Normalised MAE')
    plt.legend()
    try:
        plt.tight_layout()
    except Exception:
        pass
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
