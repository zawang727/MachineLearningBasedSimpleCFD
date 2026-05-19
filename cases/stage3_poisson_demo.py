"""
Stage 3 sanity check — finite-volume Poisson on an unstructured triangle mesh.

Solves  ∇²p = −2π² sin(πx) sin(πy)  on  Ω = [0,1]²  with  p = 0  on ∂Ω.
The analytical solution is

    p(x, y) = sin(πx) sin(πy).

KNOWN LIMITATION (Phase 3a):  this is the simplest unstructured FV stencil
— one face flux  (p_R − p_L) / d_⟂ · A.  Triangle meshes (including the
rect-triangulation here) are *non-orthogonal*: the line connecting two
cell centroids is not aligned with the face normal at most faces.  With a
single neighbour pair we cannot recover the tangential component of the
face gradient, so the discretisation is only first-order in space and
plateaus at a finite L² error as h shrinks.  The pattern of the error
(four symmetric blobs in the interior) is the signature of this
non-orthogonality, not a bug.

What we'd add to make it second-order — and the headline item for
Phase 3b — is a *non-orthogonal correction*: reconstruct ∇p at each cell
via Green-Gauss (or least-squares), then split the face flux into an
orthogonal part (this stencil) plus a deferred correction term using the
reconstructed gradient.  Standard machinery in OpenFOAM / SU2 / Basilisk.

Run:
    python cases/stage3_poisson_demo.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cfd.mesh        import rect_triangulation
from cfd.fv_poisson  import solve_poisson


def analytical(xy: np.ndarray) -> np.ndarray:
    """p(x, y) = sin(πx) sin(πy)."""
    return np.sin(np.pi * xy[..., 0]) * np.sin(np.pi * xy[..., 1])


def source(xy: np.ndarray) -> np.ndarray:
    """For ∇²p = source, source = -2π² sin(πx) sin(πy)."""
    return -2.0 * np.pi ** 2 * analytical(xy)


def all_walls_dirichlet(face_centroid, face_normal) -> dict:
    """p = 0 on every boundary face."""
    return {'type': 'dirichlet', 'value': 0.0}


def run(out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'n':>4} {'cells':>7} {'L2_err':>10} {'rate':>6}")
    prev_err = None
    prev_h   = None
    results  = []
    for n in (16, 32, 64):
        mesh = rect_triangulation(n, n)
        p    = solve_poisson(mesh, source, all_walls_dirichlet)
        ref  = analytical(mesh.cell_centroids)

        # L² error weighted by cell area.
        l2_err = np.sqrt(np.sum((p - ref) ** 2 * mesh.cell_areas))

        h = 1.0 / n
        rate = '' if prev_err is None else f"{np.log(prev_err / l2_err) / np.log(prev_h / h):.2f}"
        print(f"{n:>4} {mesh.n_cells:>7d} {l2_err:>10.4e} {rate:>6}")
        prev_err = l2_err
        prev_h   = h
        results.append((mesh, p, ref, l2_err))

    # ----- Plot the finest resolution -----
    mesh, p, ref, _ = results[-1]
    nodes = mesh.nodes
    tris  = mesh.cells

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cf0 = axes[0].tripcolor(nodes[:, 0], nodes[:, 1], tris, p,
                            shading='flat', cmap='RdBu_r',
                            vmin=-1.05, vmax=1.05)
    axes[0].set_title('FV solution p')
    axes[0].set_aspect('equal'); fig.colorbar(cf0, ax=axes[0])

    cf1 = axes[1].tripcolor(nodes[:, 0], nodes[:, 1], tris, ref,
                            shading='flat', cmap='RdBu_r',
                            vmin=-1.05, vmax=1.05)
    axes[1].set_title('Analytical sin(πx)·sin(πy)')
    axes[1].set_aspect('equal'); fig.colorbar(cf1, ax=axes[1])

    err = np.abs(p - ref)
    cf2 = axes[2].tripcolor(nodes[:, 0], nodes[:, 1], tris, err,
                            shading='flat', cmap='Reds')
    axes[2].triplot(nodes[:, 0], nodes[:, 1], tris,
                    color='black', alpha=0.08, linewidth=0.3)
    axes[2].set_title('|FV − analytical|  (mesh overlay)')
    axes[2].set_aspect('equal'); fig.colorbar(cf2, ax=axes[2])

    fig.suptitle(
        'Stage 3 sanity check: FV Poisson on a 64×64 triangulated unit square',
        fontsize=12)
    out = os.path.join(out_dir, 'stage3_poisson_demo.png')
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {out}")


if __name__ == '__main__':
    run()
