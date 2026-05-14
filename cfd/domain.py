from __future__ import annotations
import numpy as np

_EDGE_BC = {
    '#': 'no_slip',
    '-': 'lid',
    '>': 'inlet',
    '<': 'outlet',
    '^': 'inlet_v',
    'v': 'outlet_v',
}
_SOLID = '*'


class Domain:
    """
    2-D rectangular domain for MAC-grid NS solver.

    Grid: nx × ny interior cells.
      p[i, j]  pressure at cell centre            shape (nx, ny)
      u[i, j]  x-velocity at right face of (i,j)  shape (nx+1, ny)
      v[i, j]  y-velocity at top  face of (i,j)   shape (nx, ny+1)

    ASCII map orientation: top row = high y (j=ny-1), bottom = j=0.

    Symbol table:
      '#'       no-slip wall (u=v=0)
      '-'       moving horizontal lid  (+x at params['lid_u'])
      '>'       inlet  left  face  (u = params['inlet_u'], v = 0)
      '<'       outlet right face  (zero-gradient, p = 0)
      '^'       inlet  bottom face (v = params['inlet_v'])
      'v'       outlet top   face
      '*'       solid obstacle block (no-slip interior)
      ' ' '.'   fluid cell
    """

    def __init__(
        self,
        nx: int, ny: int,
        dx: float, dy: float,
        solid:     np.ndarray,    # (nx, ny) bool
        bc_type:   dict,          # 'left'/'right'/'top'/'bottom' → str
        bc_values: dict,          # 'inlet_u', 'inlet_v', 'lid_u', ...
    ) -> None:
        self.nx, self.ny   = nx, ny
        self.dx, self.dy   = dx, dy
        self.solid         = solid.astype(bool)
        self.bc_type       = bc_type
        self.bc_values     = bc_values

        # Encode BCs as spatial maps for the ML surrogate input
        self.inlet_u_map = np.zeros((nx, ny), dtype=np.float32)
        self.lid_u_map   = np.zeros((nx, ny), dtype=np.float32)

        u_in  = float(bc_values.get('inlet_u', 0.0))
        lid_u = float(bc_values.get('lid_u',   0.0))

        if bc_type.get('left')   == 'inlet':   self.inlet_u_map[0,  :] = u_in
        if bc_type.get('top')    == 'lid':     self.lid_u_map[:,  -1] = lid_u
        if bc_type.get('bottom') == 'lid':     self.lid_u_map[:,   0] = lid_u

    # ------------------------------------------------------------------
    @classmethod
    def from_ascii(
        cls,
        map_str: str,
        params:  dict,
        dx:      float = 1.0,
        dy:      float | None = None,
    ) -> Domain:
        if dy is None:
            dy = dx

        lines = [ln for ln in map_str.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("ASCII map is empty")

        width = max(len(ln) for ln in lines)
        lines = [ln.ljust(width) for ln in lines]

        # Build char grid, flip so row-0 = bottom (low y)
        grid = np.array([[c for c in ln] for ln in lines])[::-1, :]

        ny_total, nx_total = grid.shape

        def _dominant(chars):
            for ch in chars:
                if ch in _EDGE_BC:
                    return _EDGE_BC[ch]
            return 'no_slip'

        # Exclude corners (shared by two edges) to avoid # overriding > or <
        bc_type = {
            'left':   _dominant(grid[1:-1, 0]),
            'right':  _dominant(grid[1:-1, -1]),
            'bottom': _dominant(grid[0, 1:-1]),
            'top':    _dominant(grid[-1, 1:-1]),
        }

        interior = grid[1:-1, 1:-1]          # (ny-2, nx-2) after stripping border
        ny_in, nx_in = interior.shape
        # Transpose so solid[i,j] = x-column i, y-row j
        solid = (interior.T == _SOLID)

        return cls(nx_in, ny_in, dx, dy, solid, bc_type, dict(params))

    @classmethod
    def closed(
        cls,
        nx: int, ny: int,
        dx: float = 1.0,
        dy: float | None = None,
        params: dict | None = None,
    ) -> Domain:
        """Closed cavity with no inlet/outlet (lid-driven cavity)."""
        if dy is None:
            dy = dx
        bc_type = {
            'left': 'no_slip', 'right': 'no_slip',
            'bottom': 'no_slip', 'top': 'lid',
        }
        return cls(nx, ny, dx, dy,
                   np.zeros((nx, ny), dtype=bool),
                   bc_type, params or {})
