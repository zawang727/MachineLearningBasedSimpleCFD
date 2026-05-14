"""Train the U-Net CFD surrogate model."""
from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from models import build_model, save_model
from cfd.visualization import plot_training_history


def load_dataset(path: str):
    d = np.load(path, allow_pickle=True)
    X = torch.tensor(d['inputs'],  dtype=torch.float32)   # (N, 3, ny, nx)
    y = torch.tensor(d['outputs'], dtype=torch.float32)   # (N, 3, ny, nx)
    return X, y


def train(
    data_path:  str   = "data.npz",
    model_out:  str   = "model.pt",
    epochs:     int   = 100,
    batch_size: int   = 4,
    lr:         float = 1e-3,
    train_frac: float = 0.8,
    base_ch:    int   = 16,
    out_dir:    str   = "results",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X, y = load_dataset(data_path)
    n = len(X)
    n_train = max(1, int(n * train_frac))
    n_val   = n - n_train
    print(f"Dataset: {n} samples  (train={n_train}, val={n_val})  shape={tuple(X.shape)}")

    # Normalise outputs per channel by training-set max
    y_scales = torch.tensor([
        y[:n_train, c].abs().max().item() or 1.0
        for c in range(y.shape[1])
    ])
    np.save(model_out.replace('.pt', '_scales.npy'), y_scales.numpy())
    y_norm = y / y_scales[None, :, None, None]

    dataset = TensorDataset(X, y_norm)
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    model = build_model(base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(xb)
        scheduler.step()
        t_loss /= n_train
        train_losses.append(t_loss)

        if n_val > 0:
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    v_loss += criterion(pred, yb).item() * len(xb)
            v_loss /= n_val
            val_losses.append(v_loss)

        if epoch % 10 == 0 or epoch == 1:
            v_str = f"  val={v_loss:.4f}" if n_val > 0 else ""
            print(f"  Epoch {epoch:>4d}/{epochs}  train={t_loss:.4f}{v_str}")

    save_model(model, model_out)
    plot_training_history(train_losses, val_losses,
                          save_path=os.path.join(out_dir, "training_history.png"))
    print(f"Training history saved to {out_dir}/training_history.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str,   default="data.npz")
    p.add_argument("--model-out",  type=str,   default="model.pt")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--base-ch",    type=int,   default=16)
    p.add_argument("--out-dir",    type=str,   default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_path  = args.data,
        model_out  = args.model_out,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        train_frac = args.train_frac,
        base_ch    = args.base_ch,
        out_dir    = args.out_dir,
    )
