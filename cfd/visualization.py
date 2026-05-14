from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_velocity(state, title: str = "", save_path: str = "") -> None:
    """Velocity magnitude colormap + streamlines."""
    uc = state.u_cell
    vc = state.v_cell
    spd = state.speed
    nx, ny = state.domain.nx, state.domain.ny
    dx, dy = state.domain.dx, state.domain.dy

    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy

    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(x, y, spd.T, levels=20, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='Speed [m/s]')
    ax.streamplot(x, y, uc.T, vc.T, color='white', linewidth=0.6,
                  density=1.2, arrowsize=0.8)
    _draw_solid(ax, state.domain, x, y)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title or 'Velocity magnitude + streamlines')
    ax.set_aspect('equal')
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


def plot_pressure(state, title: str = "", save_path: str = "") -> None:
    nx, ny = state.domain.nx, state.domain.ny
    dx, dy = state.domain.dx, state.domain.dy
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy

    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(x, y, state.p.T, levels=20, cmap='RdBu_r')
    fig.colorbar(cf, ax=ax, label='Pressure [Pa]')
    ax.contour(x, y, state.p.T, levels=10, colors='k', linewidths=0.4, alpha=0.5)
    _draw_solid(ax, state.domain, x, y)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title or 'Pressure field')
    ax.set_aspect('equal')
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


def plot_fields(state, title: str = "", save_path: str = "") -> None:
    """1×3 panel: speed | pressure | vorticity."""
    nx, ny = state.domain.nx, state.domain.ny
    dx, dy = state.domain.dx, state.domain.dy
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy

    fields = [
        (state.speed.T,      'Speed',      'viridis',  'Speed [m/s]'),
        (state.p.T,          'Pressure',   'RdBu_r',   'Pressure [Pa]'),
        (state.vorticity.T,  'Vorticity',  'bwr',      'Vorticity [1/s]'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (data, name, cmap, cbar_label) in zip(axes, fields):
        vmax = max(np.abs(data).max(), 1e-12)
        vmin = -vmax if cmap == 'bwr' or cmap == 'RdBu_r' else 0
        cf = ax.contourf(x, y, data, levels=20, cmap=cmap,
                         vmin=vmin, vmax=vmax)
        fig.colorbar(cf, ax=ax, label=cbar_label, fraction=0.046)
        _draw_solid(ax, state.domain, x, y)
        ax.set_title(name); ax.set_xlabel('x'); ax.set_ylabel('y')
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


def plot_comparison(
    state_nn,
    state_cfd,
    title: str = "",
    save_path: str = "",
) -> None:
    """
    3×2 comparison grid.
    Rows: u_cell | v_cell | pressure
    Cols: NN prediction | CFD solver
    """
    nx, ny = state_cfd.domain.nx, state_cfd.domain.ny
    dx, dy = state_cfd.domain.dx, state_cfd.domain.dy
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy

    rows = [
        ('u-velocity',  state_nn.u_cell.T,  state_cfd.u_cell.T,  'RdBu_r'),
        ('v-velocity',  state_nn.v_cell.T,  state_cfd.v_cell.T,  'RdBu_r'),
        ('pressure',    state_nn.p.T,        state_cfd.p.T,        'RdBu_r'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    col_labels = ['NN prediction', 'CFD solver']

    for r, (name, nn_data, cfd_data, cmap) in enumerate(rows):
        vmax = max(np.abs(cfd_data).max(), 1e-12)
        vmin = -vmax
        for c, (data, col_lbl) in enumerate(zip([nn_data, cfd_data], col_labels)):
            ax = axes[r, c]
            cf = ax.contourf(x, y, data, levels=20, cmap=cmap,
                             vmin=vmin, vmax=vmax)
            fig.colorbar(cf, ax=ax, fraction=0.046)
            _draw_solid(ax, state_cfd.domain, x, y)
            if r == 0:
                ax.set_title(col_lbl, fontsize=11)
            if c == 0:
                ax.set_ylabel(name, fontsize=10)
            ax.set_aspect('equal')

    mae_u = np.mean(np.abs(state_nn.u_cell - state_cfd.u_cell))
    mae_v = np.mean(np.abs(state_nn.v_cell - state_cfd.v_cell))
    mae_p = np.mean(np.abs(state_nn.p      - state_cfd.p))
    subtitle = f'MAE  u={mae_u:.4f}  v={mae_v:.4f}  p={mae_p:.4f}'
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


def plot_training_history(train_losses: list, val_losses: list, save_path: str = "") -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train MAE')
    if val_losses:
        plt.plot(val_losses, label='Val MAE')
    plt.xlabel('Epoch'); plt.ylabel('Normalised MAE')
    plt.legend(); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def _draw_solid(ax, domain, x, y) -> None:
    """Overlay solid obstacles as grey patches."""
    if not domain.solid.any():
        return
    dx, dy = domain.dx, domain.dy
    from matplotlib.patches import Rectangle
    solid = domain.solid
    for i in range(domain.nx):
        for j in range(domain.ny):
            if solid[i, j]:
                ax.add_patch(Rectangle(
                    (i * dx, j * dy), dx, dy,
                    facecolor='grey', edgecolor='none', alpha=0.85,
                ))
