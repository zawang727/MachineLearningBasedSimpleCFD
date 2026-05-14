from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as mplcm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


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


# ------------------------------------------------------------------ #
#  True 3-D plots
# ------------------------------------------------------------------ #

def _get_field(state, name: str) -> np.ndarray:
    """Return (nx, ny, nz) array for a named field."""
    return {
        'speed':    state.speed,
        'u':        state.u_cell,
        'v':        state.v_cell,
        'w':        state.w_cell,
        'pressure': state.p,
        'p':        state.p,
    }[name]


def plot_3d_slices(
    state,
    field:     str  = 'speed',
    cmap_name: str  = 'viridis',
    symmetric: bool = False,
    title:     str  = '',
    save_path: str  = '',
) -> None:
    """
    Three orthogonal coloured slice planes (xy / xz / yz midplanes) in a
    single 3-D axes.  No 3rd-party libraries required beyond matplotlib.
    """
    dom = state.domain
    nx, ny, nz = dom.nx, dom.ny, dom.nz
    x = (np.arange(nx) + 0.5) * dom.dx
    y = (np.arange(ny) + 0.5) * dom.dy
    z = (np.arange(nz) + 0.5) * dom.dz

    data = _get_field(state, field)   # (nx, ny, nz)
    vmax = float(np.abs(data).max()) or 1e-12
    vmin = -vmax if symmetric else float(data.min())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    # ---- xy-midplane at z = z_mid ----
    kz  = nz // 2
    Xi, Yi = np.meshgrid(x, y, indexing='ij')   # (nx, ny)
    Zi  = np.full_like(Xi, z[kz])
    ax.plot_surface(Xi, Yi, Zi,
                    facecolors=cmap(norm(data[:, :, kz])),
                    shade=False, alpha=0.92, rcount=ny, ccount=nx)

    # ---- xz-midplane at y = y_mid ----
    jy  = ny // 2
    Xk, Zk = np.meshgrid(x, z, indexing='ij')   # (nx, nz)
    Yk  = np.full_like(Xk, y[jy])
    ax.plot_surface(Xk, Yk, Zk,
                    facecolors=cmap(norm(data[:, jy, :])),
                    shade=False, alpha=0.92, rcount=nz, ccount=nx)

    # ---- yz-midplane at x = x_mid ----
    ix  = nx // 2
    Yj, Zj = np.meshgrid(y, z, indexing='ij')   # (ny, nz)
    Xj  = np.full_like(Yj, x[ix])
    ax.plot_surface(Xj, Yj, Zj,
                    facecolors=cmap(norm(data[ix, :, :])),
                    shade=False, alpha=0.92, rcount=nz, ccount=ny)

    # Domain wireframe box
    for xs, ys, zs in [
        ([x[0], x[-1]], [y[0],  y[0] ], [z[0],  z[0] ]),
        ([x[0], x[-1]], [y[-1], y[-1]], [z[0],  z[0] ]),
        ([x[0], x[-1]], [y[0],  y[0] ], [z[-1], z[-1]]),
        ([x[0], x[-1]], [y[-1], y[-1]], [z[-1], z[-1]]),
        ([x[0],  x[0] ], [y[0], y[-1]], [z[0],  z[0] ]),
        ([x[-1], x[-1]], [y[0], y[-1]], [z[0],  z[0] ]),
        ([x[0],  x[0] ], [y[0], y[-1]], [z[-1], z[-1]]),
        ([x[-1], x[-1]], [y[0], y[-1]], [z[-1], z[-1]]),
        ([x[0],  x[0] ], [y[0],  y[0] ], [z[0], z[-1]]),
        ([x[-1], x[-1]], [y[0],  y[0] ], [z[0], z[-1]]),
        ([x[0],  x[0] ], [y[-1], y[-1]], [z[0], z[-1]]),
        ([x[-1], x[-1]], [y[-1], y[-1]], [z[0], z[-1]]),
    ]:
        ax.plot(xs, ys, zs, 'k-', lw=0.5, alpha=0.3)

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    sm = mplcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.1,
                 label=field.capitalize())

    if title:
        ax.set_title(title, fontsize=11)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_3d_vectors(
    state,
    stride:    int  = None,
    scale:     float = 0.6,
    title:     str  = '',
    save_path: str  = '',
) -> None:
    """
    Sparse 3-D velocity quiver arrows coloured by speed magnitude.
    """
    dom = state.domain
    nx, ny, nz = dom.nx, dom.ny, dom.nz
    x = (np.arange(nx) + 0.5) * dom.dx
    y = (np.arange(ny) + 0.5) * dom.dy
    z = (np.arange(nz) + 0.5) * dom.dz

    if stride is None:
        stride = max(1, min(nx, ny, nz) // 6)

    xi = np.arange(stride // 2, nx, stride)
    yi = np.arange(stride // 2, ny, stride)
    zi = np.arange(stride // 2, nz, stride)

    Xq, Yq, Zq = np.meshgrid(x[xi], y[yi], z[zi], indexing='ij')
    Uq = state.u_cell[np.ix_(xi, yi, zi)]
    Vq = state.v_cell[np.ix_(xi, yi, zi)]
    Wq = state.w_cell[np.ix_(xi, yi, zi)]
    mag = np.sqrt(Uq**2 + Vq**2 + Wq**2)
    mag_safe = np.where(mag > 1e-10, mag, 1.0)

    norm = Normalize(vmin=0, vmax=float(state.speed.max()) or 1.0)
    cmap = plt.get_cmap('plasma')
    colors = cmap(norm(mag)).reshape(-1, 4)

    arrow_len = stride * min(dom.dx, dom.dy, dom.dz) * scale

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.quiver(Xq.ravel(), Yq.ravel(), Zq.ravel(),
              (Uq / mag_safe).ravel(),
              (Vq / mag_safe).ravel(),
              (Wq / mag_safe).ravel(),
              length=arrow_len, colors=colors, arrow_length_ratio=0.3)

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[0], y[-1]); ax.set_zlim(z[0], z[-1])

    sm = mplcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.1, label='Speed [m/s]')

    if title:
        ax.set_title(title, fontsize=11)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_3d_case_summary(
    state,
    title:     str = '',
    save_path: str = '',
) -> None:
    """
    2-panel figure: left = speed slice planes, right = velocity quiver.
    """
    dom = state.domain
    nx, ny, nz = dom.nx, dom.ny, dom.nz
    x = (np.arange(nx) + 0.5) * dom.dx
    y = (np.arange(ny) + 0.5) * dom.dy
    z = (np.arange(nz) + 0.5) * dom.dz

    stride = max(1, min(nx, ny, nz) // 6)
    xi = np.arange(stride // 2, nx, stride)
    yi = np.arange(stride // 2, ny, stride)
    zi = np.arange(stride // 2, nz, stride)

    speed = state.speed
    vmax  = float(speed.max()) or 1.0
    norm  = Normalize(vmin=0, vmax=vmax)
    cmap  = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # --- Left: speed slice planes ---
    kz = nz // 2;  jy = ny // 2;  ix = nx // 2

    Xi, Yi = np.meshgrid(x, y, indexing='ij')
    ax1.plot_surface(Xi, Yi, np.full_like(Xi, z[kz]),
                     facecolors=cmap(norm(speed[:, :, kz])),
                     shade=False, alpha=0.9, rcount=ny, ccount=nx)

    Xk, Zk = np.meshgrid(x, z, indexing='ij')
    ax1.plot_surface(Xk, np.full_like(Xk, y[jy]), Zk,
                     facecolors=cmap(norm(speed[:, jy, :])),
                     shade=False, alpha=0.9, rcount=nz, ccount=nx)

    Yj, Zj = np.meshgrid(y, z, indexing='ij')
    ax1.plot_surface(np.full_like(Yj, x[ix]), Yj, Zj,
                     facecolors=cmap(norm(speed[ix, :, :])),
                     shade=False, alpha=0.9, rcount=nz, ccount=ny)

    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
    ax1.set_title('Speed — orthogonal midplane slices')
    sm = mplcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, shrink=0.45, pad=0.1, label='Speed [m/s]')

    # --- Right: 3-D velocity vectors ---
    Xq, Yq, Zq = np.meshgrid(x[xi], y[yi], z[zi], indexing='ij')
    Uq = state.u_cell[np.ix_(xi, yi, zi)]
    Vq = state.v_cell[np.ix_(xi, yi, zi)]
    Wq = state.w_cell[np.ix_(xi, yi, zi)]
    mag = np.sqrt(Uq**2 + Vq**2 + Wq**2)
    mag_s = np.where(mag > 1e-10, mag, 1.0)

    cmap2  = plt.get_cmap('plasma')
    colors = cmap2(norm(mag)).reshape(-1, 4)
    arrow_len = stride * min(dom.dx, dom.dy, dom.dz) * 0.55
    ax2.quiver(Xq.ravel(), Yq.ravel(), Zq.ravel(),
               (Uq / mag_s).ravel(), (Vq / mag_s).ravel(), (Wq / mag_s).ravel(),
               length=arrow_len, colors=colors, arrow_length_ratio=0.3)
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
    ax2.set_xlim(x[0], x[-1]); ax2.set_ylim(y[0], y[-1]); ax2.set_zlim(z[0], z[-1])
    ax2.set_title('Velocity vectors (normalised, coloured by speed)')
    sm2 = mplcm.ScalarMappable(cmap=cmap2, norm=norm)
    sm2.set_array([])
    fig.colorbar(sm2, ax=ax2, shrink=0.45, pad=0.1, label='Speed [m/s]')

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    try:
        plt.tight_layout()
    except Exception:
        pass
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
