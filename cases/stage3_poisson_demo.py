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
from cfd.fv_poisson  import solve_poisson, solve_poisson_2nd_order


def analytical(xy: np.ndarray) -> np.ndarray:
    """p(x, y) = sin(πx) sin(πy)."""
    return np.sin(np.pi * xy[..., 0]) * np.sin(np.pi * xy[..., 1])


def source(xy: np.ndarray) -> np.ndarray:
    """For ∇²p = source, source = -2π² sin(πx) sin(πy)."""
    return -2.0 * np.pi ** 2 * analytical(xy)


def all_walls_dirichlet(face_centroid, face_normal) -> dict:
    """p = 0 on every boundary face."""
    return {'type': 'dirichlet', 'value': 0.0}


def _refinement_table(label, solver_call, sizes):
    """Run `solver_call(mesh)` at each n in sizes and report L² + rate."""
    print(f"\n{label}")
    print(f"  {'n':>4} {'cells':>7} {'L2_err':>10} {'rate':>6}")
    out = []
    prev_err = prev_h = None
    for n in sizes:
        mesh = rect_triangulation(n, n)
        p    = solver_call(mesh)
        ref  = analytical(mesh.cell_centroids)
        l2_err = float(np.sqrt(np.sum((p - ref) ** 2 * mesh.cell_areas)))
        h      = 1.0 / n
        rate   = ('' if prev_err is None
                  else f"{np.log(prev_err / l2_err) / np.log(prev_h / h):.2f}")
        print(f"  {n:>4} {mesh.n_cells:>7d} {l2_err:>10.4e} {rate:>6}")
        prev_err, prev_h = l2_err, h
        out.append((n, mesh, p, ref, l2_err))
    return out


def run(out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)

    sizes = (16, 32, 64, 128)
    first  = _refinement_table(
        'First-order FV  (single-pair stencil, no correction)',
        lambda m: solve_poisson(m, source, all_walls_dirichlet),
        sizes)
    second = _refinement_table(
        'Second-order FV  (Green-Gauss + over-relaxed non-orth correction)',
        lambda m: solve_poisson_2nd_order(m, source, all_walls_dirichlet,
                                           max_iters=20, tol=1e-10)[0],
        sizes)

    # ---------- Refinement plot (log-log) ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    hs    = np.array([1.0 / n for n, *_ in first])
    err_1 = np.array([row[-1] for row in first])
    err_2 = np.array([row[-1] for row in second])

    ax = axes[0]
    ax.loglog(hs, err_1, 'o-', label='First-order FV')
    ax.loglog(hs, err_2, 's-', label='Second-order FV (Green-Gauss + correction)')
    # Reference slopes
    h_ref = hs
    ax.loglog(h_ref, 0.6 * h_ref ** 1,  'k--', alpha=0.4, label='O(h)  reference')
    ax.loglog(h_ref, 0.6 * h_ref ** 2,  'k:',  alpha=0.4, label='O(h²) reference')
    ax.set_xlabel('h = 1/n')
    ax.set_ylabel(r'$\|p - p_{exact}\|_{L^2}$')
    ax.set_title('Convergence on rect_triangulation')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()

    # ---------- Field plot — finest mesh, second-order solution ----------
    _, mesh, p, ref, _ = second[-1]
    nodes = mesh.nodes
    tris  = mesh.cells
    ax = axes[1]
    err = np.abs(p - ref)
    cf = ax.tripcolor(nodes[:, 0], nodes[:, 1], tris, err,
                      shading='flat', cmap='Reds')
    ax.triplot(nodes[:, 0], nodes[:, 1], tris,
               color='black', alpha=0.05, linewidth=0.2)
    fig.colorbar(cf, ax=ax)
    ax.set_title(f'|second-order − analytical|  (n={int(1 / hs[-1])})')
    ax.set_aspect('equal')

    fig.suptitle('Stage 3b: FV Poisson convergence  '
                 '(non-orthogonal correction restores O(h²))',
                 fontsize=12)
    out = os.path.join(out_dir, 'stage3_poisson_demo.png')
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {out}")


if __name__ == '__main__':
    run()
