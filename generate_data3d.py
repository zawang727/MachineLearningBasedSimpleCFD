"""
Generate 3-D CFD training dataset.

Saves a .npz file with:
  inputs  (N, 6, nz, ny, nx)  — solid, inlet_u, lid_u, dx/Lx, dy/Ly, dz/Lz
  outputs (N, 4, nz, ny, nx)  — u_cell, v_cell, w_cell, pressure
  meta    (N,)                 — JSON string per sample

Channels 3-5 are per-cell spacings normalised by total domain length so the
3-D surrogate can resolve stretched grids; on a uniform mesh they collapse
to constant maps of 1/nx, 1/ny, 1/nz.

Run:
    python generate_data3d.py --n-per-case 2 --output data3d.npz
    python generate_data3d.py --nx 32 --ny 32 --nz 32 --n-per-case 3
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

from cases import lid_driven_cavity_3d, channel_flow_3d

NX_DEFAULT = 32
NY_DEFAULT = 32
NZ_DEFAULT = 32

INPUT_CHANNELS = 6     # solid, inlet_u, lid_u, dx/Lx, dy/Ly, dz/Lz


def _encode_input(domain) -> np.ndarray:
    """(6, nz, ny, nx) float32 — solid, inlet_u, lid_u, dx/Lx, dy/Ly, dz/Lz."""
    nx, ny, nz = domain.nx, domain.ny, domain.nz
    solid   = domain.solid.astype(np.float32)          # (nx, ny, nz)
    inlet_u = domain.inlet_u_map.astype(np.float32)
    lid_u   = domain.lid_u_map.astype(np.float32)
    # Per-cell spacings normalised by domain length.
    dx_norm = (domain.dx_arr / domain.Lx).astype(np.float32)        # (nx,)
    dy_norm = (domain.dy_arr / domain.Ly).astype(np.float32)        # (ny,)
    dz_norm = (domain.dz_arr / domain.Lz).astype(np.float32)        # (nz,)
    # Broadcast each to (nz, ny, nx) — CNN convention has fastest axis = x.
    dx_map = np.broadcast_to(dx_norm[None, None, :], (nz, ny, nx)).copy()
    dy_map = np.broadcast_to(dy_norm[None, :, None], (nz, ny, nx)).copy()
    dz_map = np.broadcast_to(dz_norm[:, None, None], (nz, ny, nx)).copy()
    return np.stack([solid.T, inlet_u.T, lid_u.T, dx_map, dy_map, dz_map], axis=0)


def _encode_output(state) -> np.ndarray:
    """(4, nz, ny, nx) float32 — u, v, w, p."""
    u = state.u_cell.astype(np.float32)   # (nx, ny, nz)
    v = state.v_cell.astype(np.float32)
    w = state.w_cell.astype(np.float32)
    p = state.p.astype(np.float32)
    return np.stack([u.T, v.T, w.T, p.T], axis=0)   # (4, nz, ny, nx)


def generate(
    nx:         int = NX_DEFAULT,
    ny:         int = NY_DEFAULT,
    nz:         int = NZ_DEFAULT,
    n_per_case: int = 2,
    out_path:   str = "data3d.npz",
    out_dir:    str = "results",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    all_in, all_out, all_meta = [], [], []

    def _record(state, case_name, extra=None):
        meta_obj = {
            'case': case_name,
            'Lx':   float(state.domain.Lx),
            'Ly':   float(state.domain.Ly),
            'Lz':   float(state.domain.Lz),
        }
        if extra:
            meta_obj.update(extra)
        all_in.append(_encode_input(state.domain))
        all_out.append(_encode_output(state))
        all_meta.append(json.dumps(meta_obj))

    Re_cavity = [100, 400, 1000]
    for Re in Re_cavity[:n_per_case]:
        print(f"\n[3D generate] Lid-driven cavity  Re={Re}")
        state = lid_driven_cavity_3d.run(
            Re=Re, nx=nx, ny=ny, nz=nz, out_dir=out_dir, quiet=True)
        _record(state, 'lid_driven_cavity_3d', {'Re': Re})

    Re_channel = [50, 100, 200]
    ny_ch = max(8, ny // 2)
    nz_ch = max(8, nz // 2)
    for Re in Re_channel[:n_per_case]:
        print(f"\n[3D generate] Channel flow  Re={Re}")
        state, _ = channel_flow_3d.run(
            Re=Re, nx=nx, ny=ny_ch, nz=nz_ch, out_dir=out_dir, quiet=True)
        state = _resample_state3d(state, nx, ny, nz)
        _record(state, 'channel_flow_3d', {'Re': Re})

    inputs  = np.stack(all_in,  axis=0)
    outputs = np.stack(all_out, axis=0)
    meta    = np.array(all_meta)

    np.savez_compressed(out_path, inputs=inputs, outputs=outputs, meta=meta)
    print(f"\nSaved {len(all_in)} 3D samples to {out_path}  shape={inputs.shape}")


def _resample_state3d(state, nx_out: int, ny_out: int, nz_out: int):
    """Bilinear resize of a 3D FlowState to the target grid size."""
    from cfd.solver3d import FlowState3D
    from cfd.domain3d import Domain3D
    import scipy.ndimage

    d = state.domain
    zx = nx_out / d.nx
    zy = ny_out / d.ny
    zz = nz_out / d.nz

    u_new = scipy.ndimage.zoom(state.u_cell, (zx, zy, zz))
    v_new = scipy.ndimage.zoom(state.v_cell, (zx, zy, zz))
    w_new = scipy.ndimage.zoom(state.w_cell, (zx, zy, zz))
    p_new = scipy.ndimage.zoom(state.p,      (zx, zy, zz))
    s_new = scipy.ndimage.zoom(d.solid.astype(float), (zx, zy, zz)) > 0.5

    domain_new = Domain3D(
        nx=nx_out, ny=ny_out, nz=nz_out,
        dx=d.dx / zx, dy=d.dy / zy, dz=d.dz / zz,
        solid=s_new,
        bc_type=d.bc_type,
        bc_values=d.bc_values,
    )

    u_face = np.zeros((nx_out + 1, ny_out, nz_out))
    v_face = np.zeros((nx_out, ny_out + 1, nz_out))
    w_face = np.zeros((nx_out, ny_out, nz_out + 1))

    u_face[1:-1, :, :] = 0.5 * (u_new[:-1, :, :] + u_new[1:, :, :])
    u_face[0,    :, :] = u_new[0,  :, :]
    u_face[-1,   :, :] = u_new[-1, :, :]

    v_face[:, 1:-1, :] = 0.5 * (v_new[:, :-1, :] + v_new[:, 1:, :])
    v_face[:,  0,   :] = v_new[:, 0,  :]
    v_face[:, -1,   :] = v_new[:, -1, :]

    w_face[:, :, 1:-1] = 0.5 * (w_new[:, :, :-1] + w_new[:, :, 1:])
    w_face[:, :,  0  ] = w_new[:, :, 0]
    w_face[:, :, -1  ] = w_new[:, :, -1]

    return FlowState3D(u_face, v_face, w_face, p_new, domain_new)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nx",         type=int, default=NX_DEFAULT)
    p.add_argument("--ny",         type=int, default=NY_DEFAULT)
    p.add_argument("--nz",         type=int, default=NZ_DEFAULT)
    p.add_argument("--n-per-case", type=int, default=2,
                   help="Variants per case (max 3)")
    p.add_argument("--output",     type=str, default="data3d.npz")
    p.add_argument("--out-dir",    type=str, default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.nx, args.ny, args.nz,
             args.n_per_case, args.output, args.out_dir)
