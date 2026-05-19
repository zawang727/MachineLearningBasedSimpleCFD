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
