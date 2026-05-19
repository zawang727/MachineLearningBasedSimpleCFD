"""
Cell-centred finite-volume Poisson solver on an UnstructuredMesh2D.

Assembles  L·p = b  where L is a sparse Laplacian-of-FV-fluxes:

   For cell  c  with neighbours  c'  across face  f:
       L[c, c']  +=  |face|_f / d_{c,c'}
       L[c, c]   -=  |face|_f / d_{c,c'}

Boundary faces are handled via BC dicts keyed by a tag function that maps
a face's centroid to a label ('left', 'right', 'top', 'bottom', or any
user-defined string):

   Dirichlet (value):    contribution  −|face|/d_{c,b} · (p_c − p_b)
                         → L diagonal -= |face|/d_{c,b}; b += p_b·|face|/d_{c,b}
   Neumann   (flux):     contribution   |face| · flux_value
                         → b += −flux_value · |face|

The construction is symmetric for Dirichlet faces and assumes near-
orthogonal meshes (line cell → cell aligned with the face normal); on
the rect_triangulation it is exact up to round-off.

Intended as a Stage-3 sanity check: validate the unstructured machinery
on a problem with a closed-form solution before building incompressible
NS on top of it.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

from .mesh import UnstructuredMesh2D


BoundaryFn = Callable[[np.ndarray, np.ndarray], dict]
"""
Boundary-condition callback.  Receives:
    face_centroid:  (2,)  face midpoint xy
    face_normal:    (2,)  outward unit normal
Returns one of:
    {'type': 'dirichlet', 'value': <float>}
    {'type': 'neumann',   'flux':  <float>}   (default if not specified)
"""


def assemble_poisson(
    mesh:      UnstructuredMesh2D,
    source_fn: Callable[[np.ndarray], np.ndarray] | float,
    bc:        BoundaryFn,
):
    """
    Build (L, b) for the FV Poisson system  L p = b  with
        −∇²p = source  in Ω,   p / ∂p/∂n  prescribed on ∂Ω.

    `source_fn` may be a constant or a function evaluating at cell centroids.
    Returns (L_csc, b, dirichlet_mask).
    """
    n           = mesh.n_cells
    face_cells  = mesh.face_cells
    face_len    = mesh.face_lengths
    face_d      = mesh.face_cell_dist
    face_cen    = mesh.face_centroids
    face_norm   = mesh.face_normals
    bnd_mask    = mesh.boundary_mask

    L  = lil_matrix((n, n))
    b  = np.zeros(n, dtype=np.float64)

    # Source term: b_c = source(centroid_c) · area_c  (Poisson is −∇²p = src)
    if callable(source_fn):
        src_vals = source_fn(mesh.cell_centroids)
    else:
        src_vals = np.full(n, float(source_fn))
    b += src_vals * mesh.cell_areas

    dirichlet_mask = np.zeros(n, dtype=bool)

    # Interior faces.
    interior = ~bnd_mask
    fi = np.where(interior)[0]
    for f in fi:
        c0, c1   = int(face_cells[f, 0]), int(face_cells[f, 1])
        coeff    = face_len[f] / face_d[f]
        L[c0, c1] += coeff
        L[c1, c0] += coeff
        L[c0, c0] -= coeff
        L[c1, c1] -= coeff

    # Boundary faces.
    fb = np.where(bnd_mask)[0]
    for f in fb:
        c0    = int(face_cells[f, 0])
        spec  = bc(face_cen[f], face_norm[f])
        kind  = spec.get('type', 'neumann')
        if kind == 'dirichlet':
            val   = float(spec['value'])
            coeff = face_len[f] / face_d[f]
            L[c0, c0] -= coeff
            b[c0]     -= val * coeff
            dirichlet_mask[c0] = True
        elif kind == 'neumann':
            flux  = float(spec.get('flux', 0.0))
            b[c0] -= flux * face_len[f]
        else:
            raise ValueError(f"Unknown BC type: {kind!r}")

    # Pure-Neumann problem is singular; pin one cell to p = 0.
    if not dirichlet_mask.any():
        L[0, :]   = 0.0
        L[0, 0]   = 1.0
        b[0]      = 0.0
        dirichlet_mask[0] = True

    return L.tocsc(), b, dirichlet_mask


def solve_poisson(
    mesh:      UnstructuredMesh2D,
    source_fn: Callable[[np.ndarray], np.ndarray] | float,
    bc:        BoundaryFn,
) -> np.ndarray:
    """Assemble + LU-solve.  Returns p (shape (n_cells,))."""
    L, b, _ = assemble_poisson(mesh, source_fn, bc)
    lu = splu(L)
    return lu.solve(b)


# ====================================================================== #
# Phase 3b — Green-Gauss gradient + over-relaxed non-orthogonal correction
# ====================================================================== #

def green_gauss_gradient(
    mesh: UnstructuredMesh2D,
    p:    np.ndarray,
    bc:   BoundaryFn,
) -> np.ndarray:
    """
    Cell-centred gradient by Green-Gauss:

        ∇p_c ≈ (1/V_c) · Σ_f  p_f · n̂_{f,c} · A_f

    Face values:
      interior:  arithmetic mean of the two cell values
      boundary:  Dirichlet → boundary value; Neumann → cell value
                 (zero-gradient approximation, refined by the deferred-
                 correction outer iteration in solve_poisson_2nd_order).
    """
    n_cells   = mesh.n_cells
    n_faces   = mesh.n_faces
    fc0       = mesh.face_cells[:, 0]
    fc1       = mesh.face_cells[:, 1]
    bnd       = mesh.boundary_mask
    interior  = ~bnd
    A_f       = mesh.face_lengths
    n_hat     = mesh.face_normals      # oriented c0 → c1 (outward at bnd)

    p_f = np.empty(n_faces, dtype=np.float64)
    p_f[interior] = 0.5 * (p[fc0[interior]] + p[fc1[interior]])

    # Boundary face values.
    for f in np.where(bnd)[0]:
        spec = bc(mesh.face_centroids[f], n_hat[f])
        if spec.get('type', 'neumann') == 'dirichlet':
            p_f[f] = float(spec['value'])
        else:
            p_f[f] = p[int(fc0[f])]

    grad     = np.zeros((n_cells, 2), dtype=np.float64)
    contrib  = p_f[:, None] * n_hat * A_f[:, None]   # (n_faces, 2)
    np.add.at(grad, fc0,           contrib)
    np.add.at(grad, fc1[interior], -contrib[interior])
    grad /= mesh.cell_areas[:, None]
    return grad


def solve_poisson_2nd_order(
    mesh:      UnstructuredMesh2D,
    source_fn: Callable[[np.ndarray], np.ndarray] | float,
    bc:        BoundaryFn,
    max_iters: int   = 12,
    tol:       float = 1e-9,
    verbose:   bool  = False,
) -> tuple[np.ndarray, int]:
    """
    Second-order FV Poisson solver via deferred-correction iteration.

    Splits the face flux into an *implicit* orthogonal part (the existing
    over-relaxed stencil, in matrix L) plus an *explicit* non-orthogonal
    correction (uses the cell-centred gradient from the previous iterate
    and lives on the RHS).  Standard Jasak 1996 / Ferziger-Peric scheme.

    Per face f (interior; vector area S_f = A_f · n̂_f, cell-to-cell vec
    d = c_R − c_L, projection d_⟂ = d · n̂_f):

        S_orth = (A_f / d_⟂) · d                  ‖ d  (over-relaxed)
        S_corr = S_f − S_orth                     ⟂ d  (tangential)

    Implicit flux:  (p_R − p_L) · A_f / d_⟂        ← already in L
    Deferred flux:  ∇p_f · S_corr                  ← added to RHS each pass

    Returns (p, iters).  Iterates until either max_iters or max-norm of
    Δp falls below `tol`.
    """
    L, b0, _ = assemble_poisson(mesh, source_fn, bc)
    lu = splu(L)

    # First-order solve — initial guess.
    p = lu.solve(b0)

    # Pre-compute S_corr per face.  d_vec uses face midpoint at the
    # boundary; correction is then evaluated only on interior faces.
    fc0       = mesh.face_cells[:, 0]
    fc1       = mesh.face_cells[:, 1]
    bnd       = mesh.boundary_mask
    interior  = ~bnd
    A_f       = mesh.face_lengths
    n_hat     = mesh.face_normals
    d_perp    = mesh.face_cell_dist
    cc        = mesh.cell_centroids

    d_vec = np.zeros((mesh.n_faces, 2), dtype=np.float64)
    d_vec[interior] = cc[fc1[interior]] - cc[fc0[interior]]
    # At boundary, d_vec stays zero (we skip correction there).

    S_f    = A_f[:, None] * n_hat
    S_orth = (A_f / d_perp)[:, None] * d_vec        # zero at boundary
    S_corr = S_f - S_orth                            # equals S_f at boundary
    # Disable boundary correction by zeroing.
    S_corr[bnd] = 0.0

    for it in range(1, max_iters + 1):
        grad = green_gauss_gradient(mesh, p, bc)             # (n_cells, 2)

        # Face-averaged gradient.  (Boundary contributions are zeroed
        # by S_corr above, so the boundary entries don't matter.)
        grad_f = np.zeros((mesh.n_faces, 2), dtype=np.float64)
        grad_f[interior] = 0.5 * (grad[fc0[interior]] + grad[fc1[interior]])

        # Correction flux per face.
        corr_flux = np.einsum('fi,fi->f', grad_f, S_corr)    # (n_faces,)

        # Move to RHS: for cell c the orthogonal stencil already represents
        # flux through c's faces in the LHS; the leftover (corrected) part
        # subtracts from the source (= b).
        b = b0.copy()
        np.add.at(b, fc0,           -corr_flux)
        np.add.at(b, fc1[interior],  corr_flux[interior])

        p_new = lu.solve(b)
        delta = float(np.max(np.abs(p_new - p)))
        if verbose:
            print(f"  iter {it:>2d}  max|Δp| = {delta:.3e}")
        p = p_new
        if delta < tol:
            return p, it

    return p, max_iters
