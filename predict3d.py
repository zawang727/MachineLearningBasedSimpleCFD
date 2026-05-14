"""
Run 3-D U-Net inference and compare with CFD solver output.

For each test sample, saves:
  sample3d_XX_comparison.png  — 4×2 grid (rows=u/v/w/p, cols=NN/CFD)

Run:
    python predict3d.py --model model3d.pt --data data3d.npz --n-samples 3
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import torch

from models import load_model_3d
from cfd.solver3d import FlowState3D
from cfd.domain3d import Domain3D
from cfd.visualization3d import plot_comparison_3d


def _tensor_to_flowstate3d(tensor: np.ndarray, domain: Domain3D) -> FlowState3D:
    """Reconstruct a FlowState3D from a (4, nz, ny, nx) output array."""
    nx, ny, nz = domain.nx, domain.ny, domain.nz

    # tensor channels: (u, v, w, p) in (nz, ny, nx) order
    # transpose back to (nx, ny, nz) for internal solver convention
    uc = tensor[0].T   # (nz, ny, nx) → (nx, ny, nz)
    vc = tensor[1].T
    wc = tensor[2].T
    p  = tensor[3].T

    u_face = np.zeros((nx + 1, ny, nz))
    v_face = np.zeros((nx, ny + 1, nz))
    w_face = np.zeros((nx, ny, nz + 1))

    u_face[1:-1, :, :] = 0.5 * (uc[:-1, :, :] + uc[1:, :, :])
    u_face[0,    :, :] = uc[0,  :, :]
    u_face[-1,   :, :] = uc[-1, :, :]

    v_face[:, 1:-1, :] = 0.5 * (vc[:, :-1, :] + vc[:, 1:, :])
    v_face[:,  0,   :] = vc[:, 0,  :]
    v_face[:, -1,   :] = vc[:, -1, :]

    w_face[:, :, 1:-1] = 0.5 * (wc[:, :, :-1] + wc[:, :, 1:])
    w_face[:, :,  0  ] = wc[:, :, 0]
    w_face[:, :, -1  ] = wc[:, :, -1]

    return FlowState3D(u_face, v_face, w_face, p, domain)


def _decode_domain3d(input_tensor: np.ndarray, meta_str: str,
                     nx: int, ny: int, nz: int) -> Domain3D:
    """Reconstruct a minimal Domain3D from the input encoding + metadata."""
    meta = json.loads(meta_str)
    case = meta.get('case', 'unknown')
    Re   = meta.get('Re', 100.0)

    # input channels in (nz, ny, nx) → transpose to (nx, ny, nz)
    solid     = input_tensor[0].T > 0.5
    inlet_u_val = float(input_tensor[1].max())
    lid_u_val   = float(input_tensor[2].max())

    dx = 1.0 / nx
    dy = 1.0 / ny
    dz = 1.0 / nz

    if case == 'lid_driven_cavity_3d':
        bc_type = {
            'left': 'no_slip', 'right': 'no_slip',
            'bottom': 'no_slip', 'top': 'lid',
            'front': 'no_slip', 'back': 'no_slip',
        }
        bc_values = {'lid_u': lid_u_val, 'rho': 1.0, 'nu': 1.0 / Re}
    elif case == 'channel_flow_3d':
        bc_type = {
            'left': 'inlet', 'right': 'outlet',
            'bottom': 'no_slip', 'top': 'no_slip',
            'front': 'no_slip', 'back': 'no_slip',
        }
        bc_values = {'inlet_u': inlet_u_val, 'rho': 1.0, 'nu': 1.0 / Re}
    else:
        bc_type = {
            'left': 'inlet', 'right': 'outlet',
            'bottom': 'no_slip', 'top': 'no_slip',
            'front': 'no_slip', 'back': 'no_slip',
        }
        bc_values = {'inlet_u': inlet_u_val, 'rho': 1.0, 'nu': 1.0 / Re}

    return Domain3D(
        nx=nx, ny=ny, nz=nz,
        dx=dx, dy=dy, dz=dz,
        solid=solid,
        bc_type=bc_type,
        bc_values=bc_values,
    )


def predict(
    model_path: str = "model3d.pt",
    data_path:  str = "data3d.npz",
    scales_path: str = "",
    n_samples:  int = 3,
    train_frac: float = 0.85,
    base_ch:    int = 8,
    out_dir:    str = "results",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if not scales_path:
        scales_path = os.path.join(os.path.dirname(model_path) or out_dir, 'scales3d.npy')
        if not os.path.exists(scales_path):
            scales_path = os.path.join(out_dir, 'scales3d.npy')

    scales = np.load(scales_path)   # (1, 4, 1, 1, 1)
    scales = scales.reshape(4)      # (4,) for broadcast

    model = load_model_3d(model_path, base_ch)
    model.eval()

    d = np.load(data_path, allow_pickle=True)
    X    = d['inputs']    # (N, 3, nz, ny, nx)
    y    = d['outputs']   # (N, 4, nz, ny, nx)
    meta = d['meta']

    n = len(X)
    split = max(1, int(n * train_frac))
    X_test    = X[split:]
    y_test    = y[split:]
    meta_test = meta[split:]

    if len(X_test) == 0:
        print("No test samples (train_frac too high). Showing training samples.")
        X_test    = X[:n_samples]
        y_test    = y[:n_samples]
        meta_test = meta[:n_samples]

    _, _, nz_grid, ny_grid, nx_grid = X_test.shape

    with torch.no_grad():
        X_t   = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_t).numpy()   # (N_test, 4, nz, ny, nx)

    # Denormalise
    preds *= scales[None, :, None, None, None]

    mae_u = np.mean(np.abs(preds[:, 0] - y_test[:, 0]))
    mae_v = np.mean(np.abs(preds[:, 1] - y_test[:, 1]))
    mae_w = np.mean(np.abs(preds[:, 2] - y_test[:, 2]))
    mae_p = np.mean(np.abs(preds[:, 3] - y_test[:, 3]))
    print(f"Overall Test MAE - u={mae_u:.4f}  v={mae_v:.4f}  "
          f"w={mae_w:.4f}  p={mae_p:.4f}")
    print(f"Saving {min(n_samples, len(X_test))} comparison images to {out_dir}/")

    for i in range(min(n_samples, len(X_test))):
        meta_i = str(meta_test[i])
        domain = _decode_domain3d(X_test[i], meta_i, nx_grid, ny_grid, nz_grid)

        state_nn  = _tensor_to_flowstate3d(preds[i],  domain)
        state_cfd = _tensor_to_flowstate3d(y_test[i], domain)

        sample_mae_u = np.mean(np.abs(preds[i, 0] - y_test[i, 0]))
        title = f"3D Sample {i} | {meta_i} | MAE_u={sample_mae_u:.4f}"

        comp_path = os.path.join(out_dir, f"sample3d_{i:02d}_comparison.png")
        plot_comparison_3d(state_nn, state_cfd, title=title, save_path=comp_path)
        print(f"  Saved {comp_path}  (MAE_u={sample_mae_u:.4f})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       type=str,   default="model3d.pt")
    p.add_argument("--data",        type=str,   default="data3d.npz")
    p.add_argument("--scales",      type=str,   default="",
                   help="Path to scales3d.npy (auto-detected if omitted)")
    p.add_argument("--n-samples",   type=int,   default=3)
    p.add_argument("--train-frac",  type=float, default=0.85)
    p.add_argument("--base-ch",     type=int,   default=8)
    p.add_argument("--out-dir",     type=str,   default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.model, args.data, args.scales, args.n_samples,
            args.train_frac, args.base_ch, args.out_dir)
