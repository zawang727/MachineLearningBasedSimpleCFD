"""
Unstructured 2-D triangle mesh — data structures and geometry helpers.

Stage-3 foundation: replaces the (nx, ny) structured-grid view of the
existing solver with explicit cell / face / node tables.  Conventions:

  nodes        (n_nodes, 2)  float  (x, y) coordinates
  cells        (n_cells, 3)  int    node indices, CCW order
  face_nodes   (n_faces, 2)  int    two node indices per face
  face_cells   (n_faces, 2)  int    adjacent cell indices, second is −1 for
                                    boundary faces (only one neighbour)

Geometry properties (cached on first access):
  cell_areas       (n_cells,)         signed triangle area (CCW > 0)
  cell_centroids   (n_cells, 2)       centroid of each triangle
  face_lengths     (n_faces,)         edge length
  face_centroids   (n_faces, 2)       face midpoint
  face_normals     (n_faces, 2)       unit normal pointing from cell 0 → cell 1
                                      (or outward at the boundary)
  face_cell_dist   (n_faces,)         distance between the two adjacent cell
                                      centroids (cell 0 → boundary-face
                                      midpoint when face_cells[:,1] == −1)

This file is intentionally NumPy-only; no scipy.sparse imports here so it
can be used by a future graph-NN data pipeline without dragging the solver
in.  Solver assembly lives in cfd.fv_poisson and (eventually) cfd.fv_solver.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class UnstructuredMesh2D:
    nodes: np.ndarray              # (n_nodes, 2)  float
    cells: np.ndarray              # (n_cells, 3)  int, CCW

    _faces: tuple | None = field(default=None, init=False, repr=False)
    _geom:  dict       | None = field(default=None, init=False, repr=False)

    @property
    def n_nodes(self) -> int: return int(self.nodes.shape[0])
    @property
    def n_cells(self) -> int: return int(self.cells.shape[0])

    # ------------------------------------------------------------------ #
    # Face connectivity
    # ------------------------------------------------------------------ #

    def _build_faces(self) -> None:
        """Pair triangle edges into faces; identify boundary faces."""
        edges: dict[tuple[int, int], list[int]] = {}
        for c in range(self.n_cells):
            v = self.cells[c]
            for i in range(3):
                key = tuple(sorted((int(v[i]), int(v[(i + 1) % 3]))))
                edges.setdefault(key, []).append(c)

        n_faces      = len(edges)
        face_nodes   = np.zeros((n_faces, 2), dtype=np.int64)
        face_cells   = np.full((n_faces, 2), -1, dtype=np.int64)
        for f, (edge, owners) in enumerate(edges.items()):
            face_nodes[f] = edge
            for i, c in enumerate(owners):
                face_cells[f, i] = c

        self._faces = (face_nodes, face_cells)

    @property
    def face_nodes(self) -> np.ndarray:
        if self._faces is None:
            self._build_faces()
        return self._faces[0]

    @property
    def face_cells(self) -> np.ndarray:
        if self._faces is None:
            self._build_faces()
        return self._faces[1]

    @property
    def n_faces(self) -> int:
        return int(self.face_nodes.shape[0])

    @property
    def boundary_mask(self) -> np.ndarray:
        """True where the face has only one adjacent cell."""
        return self.face_cells[:, 1] == -1

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #

    def _compute_geometry(self) -> None:
        v = self.nodes[self.cells]                       # (n_cells, 3, 2)
        v0, v1, v2 = v[:, 0], v[:, 1], v[:, 2]

        # Signed area (CCW positive); absolute used for volumes.
        signed   = 0.5 * ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
                         - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
        cell_area      = np.abs(signed)
        cell_centroid  = v.mean(axis=1)

        fn   = self.face_nodes
        n0   = self.nodes[fn[:, 0]]
        n1   = self.nodes[fn[:, 1]]
        edge = n1 - n0                                   # (n_faces, 2)
        face_length    = np.linalg.norm(edge, axis=1)
        face_centroid  = 0.5 * (n0 + n1)

        # Unit normal: 90° rotation of the edge.  Orient from cell 0 → cell 1
        # at interior faces, outward at boundary faces.
        raw_normal = np.column_stack([ edge[:, 1], -edge[:, 0]])
        unit_normal = raw_normal / face_length[:, None]

        face_cells = self.face_cells
        c0   = face_cells[:, 0]
        c1   = face_cells[:, 1]
        cen0 = cell_centroid[c0]
        bnd  = (c1 == -1)
        cen1 = np.where(bnd[:, None], face_centroid, cell_centroid[np.where(bnd, c0, c1)])

        # Flip normal if it points from cell 1 back toward cell 0 (or inward
        # at boundary).
        dir_vec   = cen1 - cen0
        flip      = np.einsum('fi,fi->f', dir_vec, unit_normal) < 0
        unit_normal[flip] = -unit_normal[flip]

        # Effective stencil distance: project the centroid-to-centroid
        # vector onto the face normal.  For an orthogonal mesh this equals
        # the Euclidean separation; for non-orthogonal cells (most triangle
        # meshes at the boundary) it is the perpendicular distance from the
        # cell centroid to the face plane, which is what the FV gradient
        # needs.  np.maximum guards against degenerate negative projections
        # (extremely skewed triangles).
        face_cell_dist = np.maximum(
            np.einsum('fi,fi->f', dir_vec, unit_normal),
            1e-14,
        )

        self._geom = dict(
            cell_areas     = cell_area,
            cell_centroids = cell_centroid,
            face_lengths   = face_length,
            face_centroids = face_centroid,
            face_normals   = unit_normal,
            face_cell_dist = face_cell_dist,
        )

    def _g(self, key: str) -> np.ndarray:
        if self._geom is None:
            self._compute_geometry()
        return self._geom[key]

    @property
    def cell_areas(self):     return self._g('cell_areas')
    @property
    def cell_centroids(self): return self._g('cell_centroids')
    @property
    def face_lengths(self):   return self._g('face_lengths')
    @property
    def face_centroids(self): return self._g('face_centroids')
    @property
    def face_normals(self):   return self._g('face_normals')
    @property
    def face_cell_dist(self): return self._g('face_cell_dist')


# ---------------------------------------------------------------------- #
# Synthetic mesh generators
# ---------------------------------------------------------------------- #

def rect_triangulation(nx: int, ny: int,
                        Lx: float = 1.0, Ly: float = 1.0
                        ) -> UnstructuredMesh2D:
    """
    Uniform triangulation of [0, Lx] × [0, Ly]: each Cartesian rectangle is
    split into two triangles along its lower-left ↗ upper-right diagonal.

    Produces (nx+1)·(ny+1) nodes and 2·nx·ny triangles — useful as a
    near-orthogonal sanity case where the FV stencil should match the
    structured solver to round-off.
    """
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    nodes  = np.column_stack([XX.ravel(), YY.ravel()])

    def nid(i, j): return i * (ny + 1) + j

    cells = []
    for i in range(nx):
        for j in range(ny):
            sw, se = nid(i,     j),     nid(i + 1, j)
            nw, ne = nid(i,     j + 1), nid(i + 1, j + 1)
            cells.append([sw, se, ne])    # lower-right triangle, CCW
            cells.append([sw, ne, nw])    # upper-left  triangle, CCW
    return UnstructuredMesh2D(nodes=nodes, cells=np.asarray(cells, dtype=np.int64))
