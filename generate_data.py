"""
Generate CFD training dataset by running all three cases at varied Re and geometry.

Saves a .npz file with:
  inputs  (N, 5, ny, nx)  — solid, inlet_u, lid_u, dx/Lx, dy/Ly  (mesh-aware)
  outputs (N, 3, ny, nx)  — u_cell, v_cell, pressure
  meta    (N,)            — JSON string per sample (case, Re, ...)

Channels 3-4 carry per-cell spacing normalised by the total domain length so
the surrogate can resolve stretched grids; on a uniform mesh they are
constant maps equal to 1/nx and 1/ny respectively.
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np

from cases import lid_driven_cavity, channel_flow, flow_around_block


# Standard grid size for all cases (must match at training time)
NX_DEFAULT = 64
NY_DEFAULT = 64

INPUT_CHANNELS = 5     # solid, inlet_u, lid_u, dx/Lx, dy/Ly


def _encode_input(domain) -> np.ndarray:
    """Build 5-channel (5, ny, nx) input tensor from Domain."""
    nx, ny  = domain.nx, domain.ny
    solid   = domain.solid.astype(np.float32)
    inlet_u = domain.inlet_u_map.astype(np.float32)
    lid_u   = domain.lid_u_map.astype(np.float32)
    # Per-cell spacings normalised by total domain length → dimensionless.
    dx_norm = (domain.dx_arr / domain.Lx).astype(np.float32)        # (nx,)
    dy_norm = (domain.dy_arr / domain.Ly).astype(np.float32)        # (ny,)
    dx_map  = np.broadcast_to(dx_norm[None, :], (ny, nx)).copy()    # (ny, nx)
    dy_map  = np.broadcast_to(dy_norm[:, None], (ny, nx)).copy()    # (ny, nx)
    return np.stack([solid.T, inlet_u.T, lid_u.T, dx_map, dy_map], axis=0)


def _encode_output(state) -> np.ndarray:
    """Build 3-channel (3, ny, nx) output tensor from FlowState."""
    uc = state.u_cell.astype(np.float32)   # (nx, ny)
    vc = state.v_cell.astype(np.float32)
    p  = state.p.astype(np.float32)
    return np.stack([uc.T, vc.T, p.T], axis=0)


def generate(
    nx:          int   = NX_DEFAULT,
    ny:          int   = NY_DEFAULT,
    n_per_case:  int   = 3,
    out_path:    str   = "data.npz",
    out_dir:     str   = "results",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    all_in, all_out, all_meta = [], [], []

    def _record(state, case_name: str, extra: dict | None = None):
        meta_obj = {
            'case': case_name,
            'Lx':   float(state.domain.Lx),
            'Ly':   float(state.domain.Ly),
        }
        if extra:
            meta_obj.update(extra)
        all_in.append(_encode_input(state.domain))
        all_out.append(_encode_output(state))
        all_meta.append(json.dumps(meta_obj))

    # ---- Lid-driven cavity ----
    Re_cavity = [100, 400, 1000]
    for Re in Re_cavity[:n_per_case]:
        print(f"\n[generate] Lid-driven cavity  Re={Re}")
        state = lid_driven_cavity.run(Re=Re, nx=nx, ny=ny, out_dir=out_dir, quiet=True)
        _record(state, 'lid_driven_cavity', {'Re': Re})

    # ---- Channel flow ----
    Re_channel = [50, 100, 200]
    for Re in Re_channel[:n_per_case]:
        print(f"\n[generate] Channel flow  Re={Re}")
        state, _ = channel_flow.run(Re=Re, nx=nx, ny=ny // 4, out_dir=out_dir, quiet=True)
        state = _resample_state(state, nx, ny)
        _record(state, 'channel_flow', {'Re': Re})

    # ---- Flow around block ----
    Re_block = [50, 100, 200]
    block_y_fracs = [0.3, 0.5, 0.7]
    count = 0
    for Re, by in zip(Re_block, block_y_fracs):
        if count >= n_per_case:
            break
        print(f"\n[generate] Flow around block  Re={Re}  block_y={by}")
        state = flow_around_block.run(Re=Re, nx=nx, ny=ny // 2,
                                      block_y_frac=by, out_dir=out_dir, quiet=True)
        state = _resample_state(state, nx, ny)
        _record(state, 'flow_around_block', {'Re': Re, 'block_y': by})
        count += 1

    inputs  = np.stack(all_in,  axis=0)   # (N, 3, ny, nx)
    outputs = np.stack(all_out, axis=0)
    meta    = np.array(all_meta)

    np.savez_compressed(out_path, inputs=inputs, outputs=outputs, meta=meta)
    print(f"\nSaved {len(all_in)} samples to {out_path}  shape={inputs.shape}")


def _resample_state(state, nx_out: int, ny_out: int):
    """Bilinear resize of FlowState fields to target grid size."""
    from cfd.solver import FlowState
    from cfd.domain import Domain
    import scipy.ndimage

    zoom_x = nx_out / state.domain.nx
    zoom_y = ny_out / state.domain.ny

    u_new = scipy.ndimage.zoom(state.u_cell, (zoom_x, zoom_y))
    v_new = scipy.ndimage.zoom(state.v_cell, (zoom_x, zoom_y))
    p_new = scipy.ndimage.zoom(state.p,      (zoom_x, zoom_y))
    s_new = scipy.ndimage.zoom(state.domain.solid.astype(float), (zoom_x, zoom_y)) > 0.5

    domain_new = Domain(
        nx=nx_out, ny=ny_out,
        dx=state.domain.dx / zoom_x,
        dy=state.domain.dy / zoom_y,
        solid=s_new,
        bc_type=state.domain.bc_type,
        bc_values=state.domain.bc_values,
    )
    # Wrap cell-centre values back into face arrays via linear interpolation
    u_face = np.zeros((nx_out + 1, ny_out))
    v_face = np.zeros((nx_out, ny_out + 1))
    u_face[1:-1, :] = 0.5 * (u_new[:-1, :] + u_new[1:, :])
    u_face[0,    :] = u_new[0,  :]
    u_face[-1,   :] = u_new[-1, :]
    v_face[:, 1:-1] = 0.5 * (v_new[:, :-1] + v_new[:, 1:])
    v_face[:,  0  ] = v_new[:, 0]
    v_face[:, -1  ] = v_new[:, -1]
    return FlowState(u_face, v_face, p_new, domain_new)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nx",          type=int, default=NX_DEFAULT)
    p.add_argument("--ny",          type=int, default=NY_DEFAULT)
    p.add_argument("--n-per-case",  type=int, default=3,
                   help="Re variants per case (max 3)")
    p.add_argument("--output",      type=str, default="data.npz")
    p.add_argument("--out-dir",     type=str, default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.nx, args.ny, args.n_per_case, args.output, args.out_dir)
