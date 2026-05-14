"""
Train 3-D U-Net surrogate.

Run:
    python train3d.py --data data3d.npz --epochs 80 --model-out model3d.pt
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from models import build_model_3d, save_model_3d
from cfd.visualization3d import plot_training_history_3d


def train(
    data_path:  str = "data3d.npz",
    epochs:     int = 80,
    batch_size: int = 2,
    lr:         float = 1e-3,
    model_out:  str = "model3d.pt",
    base_ch:    int = 8,
    out_dir:    str = "results",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    data     = np.load(data_path)
    inputs_  = torch.from_numpy(data['inputs'].astype(np.float32))   # (N,3,nz,ny,nx)
    outputs_ = torch.from_numpy(data['outputs'].astype(np.float32))  # (N,4,nz,ny,nx)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: {len(inputs_)} samples  shape={tuple(inputs_.shape)}")

    # Normalise each output channel by its training-set maximum
    n_val   = max(1, int(0.15 * len(inputs_)))
    n_train = len(inputs_) - n_val
    scales  = outputs_[:n_train].abs().amax(dim=(0, 2, 3, 4), keepdim=True)  # (1,4,1,1,1)
    scales  = scales.clamp(min=1e-6)
    np.save(os.path.join(out_dir, 'scales3d.npy'), scales.numpy())
    outputs_norm = outputs_ / scales

    dataset    = TensorDataset(inputs_, outputs_norm)
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_dl   = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl     = DataLoader(val_ds,   batch_size=batch_size)

    model = build_model_3d(base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"3D U-Net parameters: {n_params:,}")

    optim    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    criterion = nn.L1Loss()

    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        tloss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
            tloss += loss.item()
        tloss /= len(train_dl)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                vloss += criterion(model(xb.to(device)), yb.to(device)).item()
        vloss /= max(1, len(val_dl))
        scheduler.step()

        train_losses.append(tloss)
        val_losses.append(vloss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{epochs}  train={tloss:.4f}  val={vloss:.4f}")

    save_model_3d(model, model_out)
    plot_training_history_3d(train_losses, val_losses,
                              save_path=os.path.join(out_dir, 'training_history_3d.png'))
    print(f"Training history saved to {out_dir}/training_history_3d.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str,   default="data3d.npz")
    p.add_argument("--epochs",     type=int,   default=80)
    p.add_argument("--batch-size", type=int,   default=2)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--model-out",  type=str,   default="model3d.pt")
    p.add_argument("--base-ch",    type=int,   default=8)
    p.add_argument("--out-dir",    type=str,   default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.epochs, args.batch_size, args.lr,
          args.model_out, args.base_ch, args.out_dir)
