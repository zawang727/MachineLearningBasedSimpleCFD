"""
Run U-Net inference and compare with CFD solver output.

For each test sample, saves:
  sample_XX_comparison.png  — 3×2 grid (rows=u/v/p, cols=NN/CFD)
  sample_XX_fields.png      — side-by-side field overview
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import torch

from models import load_model
from cfd.solver import FlowState
from cfd.domain import Domain
from cfd.visualization import plot_comparison, plot_fields


def _tensor_to_flowstate(tensor: np.ndarray, domain: Domain) -> FlowState:
    """Reconstruct a FlowState from a (3, ny, nx) output array."""
    nx, ny = domain.nx, domain.ny
    uc = tensor[0].T   # (ny, nx) → (nx, ny)
    vc = tensor[1].T
    p  = tensor[2].T

    u_face = np.zeros((nx + 1, ny))
    v_face = np.zeros((nx, ny + 1))
    u_face[1:-1, :] = 0.5 * (uc[:-1, :] + uc[1:, :])
    u_face[0,    :] = uc[0,  :];  u_face[-1, :] = uc[-1, :]
    v_face[:, 1:-1] = 0.5 * (vc[:, :-1] + vc[:, 1:])
    v_face[:,  0  ] = vc[:, 0];   v_face[:, -1] = vc[:, -1]
    return FlowState(u_face, v_face, p, domain)


def _decode_domain(input_tensor: np.ndarray, meta_str: str,
                   nx: int, ny: int) -> Domain:
    """Reconstruct a minimal Domain from the input encoding + metadata."""
    meta = json.loads(meta_str)
    case = meta.get('case', 'unknown')
    Re   = meta.get('Re', 100.0)

    solid = input_tensor[0].T > 0.5   # (nx, ny)
    inlet_u_val = float(input_tensor[1].max())
    lid_u_val   = float(input_tensor[2].max())

    if case == 'lid_driven_cavity':
        bc_type = {'left': 'no_slip', 'right': 'no_slip',
                   'bottom': 'no_slip', 'top': 'lid'}
        bc_values = {'lid_u': lid_u_val, 'rho': 1.0, 'nu': 1.0 / Re}
    elif case == 'channel_flow':
        bc_type = {'left': 'inlet', 'right': 'outlet',
                   'bottom': 'no_slip', 'top': 'no_slip'}
        bc_values = {'inlet_u': inlet_u_val, 'rho': 1.0, 'nu': 1.0 / Re}
    else:
        bc_type = {'left': 'inlet', 'right': 'outlet',
                   'bottom': 'no_slip', 'top': 'no_slip'}
        bc_values = {'inlet_u': inlet_u_val, 'rho': 1.0, 'nu': 1.0 / Re}

    return Domain(nx=nx, ny=ny, dx=1.0/nx, dy=1.0/ny,
                  solid=solid, bc_type=bc_type, bc_values=bc_values)


def predict(
    model_path: str = "model.pt",
    data_path:  str = "data.npz",
    n_samples:  int = 3,
    train_frac: float = 0.8,
    base_ch:    int = 16,
    out_dir:    str = "results",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    scales = np.load(model_path.replace('.pt', '_scales.npy'))
    model  = load_model(model_path, base_ch)
    model.eval()

    d = np.load(data_path, allow_pickle=True)
    X    = d['inputs']    # (N, 3, ny, nx)
    y    = d['outputs']   # (N, 3, ny, nx)
    meta = d['meta']

    n = len(X)
    split = max(1, int(n * train_frac))
    X_test    = X[split:]
    y_test    = y[split:]
    meta_test = meta[split:]

    if len(X_test) == 0:
        print("No test samples (train_frac too high). Showing training samples.")
        X_test = X[:n_samples];  y_test = y[:n_samples];  meta_test = meta[:n_samples]

    _, _, ny_grid, nx_grid = X_test.shape

    # Run inference
    with torch.no_grad():
        X_t   = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_t).numpy()   # (N_test, 3, ny, nx)

    # Denormalise
    preds *= scales[None, :, None, None]

    mae_u = np.mean(np.abs(preds[:, 0] - y_test[:, 0]))
    mae_v = np.mean(np.abs(preds[:, 1] - y_test[:, 1]))
    mae_p = np.mean(np.abs(preds[:, 2] - y_test[:, 2]))
    print(f"Overall Test MAE - u={mae_u:.4f}  v={mae_v:.4f}  p={mae_p:.4f}")
    print(f"Saving {min(n_samples, len(X_test))} comparison images to {out_dir}/")

    for i in range(min(n_samples, len(X_test))):
        meta_i = str(meta_test[i])
        domain = _decode_domain(X_test[i], meta_i, nx_grid, ny_grid)

        state_nn  = _tensor_to_flowstate(preds[i],   domain)
        state_cfd = _tensor_to_flowstate(y_test[i],  domain)

        sample_mae_u = np.mean(np.abs(preds[i, 0] - y_test[i, 0]))
        title = f"Sample {i} | {meta_i} | MAE_u={sample_mae_u:.4f}"

        comp_path   = os.path.join(out_dir, f"sample_{i:02d}_comparison.png")
        fields_path = os.path.join(out_dir, f"sample_{i:02d}_fields_nn.png")

        plot_comparison(state_nn, state_cfd, title=title, save_path=comp_path)
        plot_fields(state_nn, title=f"NN prediction — {meta_i}", save_path=fields_path)
        print(f"  Saved {comp_path}  (MAE_u={sample_mae_u:.4f})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       type=str,   default="model.pt")
    p.add_argument("--data",        type=str,   default="data.npz")
    p.add_argument("--n-samples",   type=int,   default=3)
    p.add_argument("--train-frac",  type=float, default=0.8)
    p.add_argument("--base-ch",     type=int,   default=16)
    p.add_argument("--out-dir",     type=str,   default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.model, args.data, args.n_samples,
            args.train_frac, args.base_ch, args.out_dir)
