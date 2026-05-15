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


def _as_spacing(value, n: int, name: str) -> np.ndarray:
    """Normalise scalar / 1-D array spacing input to a positive (n,) ndarray."""
    if np.isscalar(value):
        return np.full(n, float(value), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"{name} array must have shape ({n},), got {arr.shape}")
    if (arr <= 0).any():
        raise ValueError(f"{name} must be strictly positive")
    return arr


def _scalar_summary(arr: np.ndarray) -> float:
    """Single-value spacing accessor — uniform value if uniform, else mean."""
    if np.allclose(arr, arr[0]):
        return float(arr[0])
    return float(arr.mean())


def tanh_spacing(n: int, L: float = 1.0, beta: float = 2.5) -> np.ndarray:
    """
    Double-sided tanh-stretched cell widths that cluster cells near *both*
    domain edges (typical use: refine near no-slip walls).

    n      : number of cells.
    L      : total length to span.
    beta   : clustering strength.  beta ≤ ~1 returns a near-uniform grid;
             larger beta gives stronger clustering at the two ends.

    Returns: dx_arr shape (n,) with sum == L.
    """
    if beta <= 1.0 + 1e-9:
        return np.full(n, L / n, dtype=float)
    s      = np.arange(n + 1, dtype=float) / n          # uniform [0, 1]
    x_face = 0.5 * L * (1.0 + np.tanh(beta * (s - 0.5)) / np.tanh(beta * 0.5))
    x_face[0]  = 0.0
    x_face[-1] = L
    return np.diff(x_face)


def _parse_axis(spec: str, n: int, L: float) -> np.ndarray:
    """
    Parse a stretching spec into a (n,) cell-width array.

    Accepted forms:
      'uniform'                  → L/n everywhere
      'tanh' / 'tanh beta=2.5'   → double-sided tanh clustering
      'list 0.02 0.03 0.04 ...'  → explicit cell widths (must sum to ≈ L)
    """
    parts = spec.strip().split()
    if not parts:
        return np.full(n, L / n)
    kind = parts[0]
    if kind == 'uniform':
        return np.full(n, L / n)
    if kind == 'tanh':
        kwargs = dict(t.split('=', 1) for t in parts[1:] if '=' in t)
        beta   = float(kwargs.get('beta', 2.5))
        return tanh_spacing(n, L, beta)
    if kind == 'list':
        widths = np.asarray([float(t) for t in parts[1:]], dtype=float)
        if widths.size != n:
            raise ValueError(f"axis 'list' expected {n} widths, got {widths.size}")
        if not np.isclose(widths.sum(), L, rtol=1e-3):
            raise ValueError(
                f"axis 'list' widths sum to {widths.sum():.6f}, expected {L}")
        return widths
    raise ValueError(f"Unknown axis kind: {kind!r}")


def _parse_mesh_ascii(map_str: str):
    """
    Parse an ASCII cell map into (nx, ny, solid, bc_type).

    Shared between Domain.from_ascii and Domain.from_text so the two
    entry points use identical mesh semantics.
    """
    lines = [ln for ln in map_str.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("ASCII map is empty")
    width = max(len(ln) for ln in lines)
    lines = [ln.ljust(width) for ln in lines]
    grid  = np.array([[c for c in ln] for ln in lines])[::-1, :]

    def _dominant(chars):
        for ch in chars:
            if ch in _EDGE_BC:
                return _EDGE_BC[ch]
        return 'no_slip'

    bc_type = {
        'left':   _dominant(grid[1:-1, 0]),
        'right':  _dominant(grid[1:-1, -1]),
        'bottom': _dominant(grid[0,    1:-1]),
        'top':    _dominant(grid[-1,   1:-1]),
    }
    interior     = grid[1:-1, 1:-1]
    ny_in, nx_in = interior.shape
    solid        = (interior.T == _SOLID)
    return nx_in, ny_in, solid, bc_type


class Domain:
    """
    2-D rectangular domain for MAC-grid NS solver.

    Grid: nx × ny interior cells.
      p[i, j]  pressure at cell centre            shape (nx, ny)
      u[i, j]  x-velocity at right face of (i,j)  shape (nx+1, ny)
      v[i, j]  y-velocity at top  face of (i,j)   shape (nx, ny+1)

    Spacing: `dx` and `dy` may be a scalar (uniform mesh) or a 1-D array of
    shape (nx,) / (ny,) for a stretched mesh.  The Domain always exposes the
    per-axis spacing arrays (`dx_arr`, `dy_arr`) and cell / face coordinate
    arrays (`x_cell`, `y_cell`, `x_face`, `y_face`) so downstream code can
    work the same way on uniform and stretched grids.

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
        dx: 'float | np.ndarray',
        dy: 'float | np.ndarray',
        solid:     np.ndarray,    # (nx, ny) bool
        bc_type:   dict,          # 'left'/'right'/'top'/'bottom' → str
        bc_values: dict,          # 'inlet_u', 'inlet_v', 'lid_u', ...
    ) -> None:
        self.nx, self.ny   = nx, ny
        self.dx_arr        = _as_spacing(dx, nx, 'dx')   # (nx,)
        self.dy_arr        = _as_spacing(dy, ny, 'dy')   # (ny,)
        self.dx            = _scalar_summary(self.dx_arr)  # scalar accessor
        self.dy            = _scalar_summary(self.dy_arr)
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
    # Mesh geometry — face positions, cell centres, total lengths.
    # All derived from dx_arr / dy_arr so they are correct on stretched grids.
    @property
    def x_face(self) -> np.ndarray:
        """u-face x positions (cell vertical edges), shape (nx+1,)."""
        return np.concatenate(([0.0], np.cumsum(self.dx_arr)))

    @property
    def y_face(self) -> np.ndarray:
        """v-face y positions (cell horizontal edges), shape (ny+1,)."""
        return np.concatenate(([0.0], np.cumsum(self.dy_arr)))

    @property
    def x_cell(self) -> np.ndarray:
        """Cell-centre x positions, shape (nx,)."""
        xf = self.x_face
        return 0.5 * (xf[:-1] + xf[1:])

    @property
    def y_cell(self) -> np.ndarray:
        """Cell-centre y positions, shape (ny,)."""
        yf = self.y_face
        return 0.5 * (yf[:-1] + yf[1:])

    @property
    def Lx(self) -> float:
        return float(self.dx_arr.sum())

    @property
    def Ly(self) -> float:
        return float(self.dy_arr.sum())

    # ------------------------------------------------------------------
    @classmethod
    def from_ascii(
        cls,
        map_str: str,
        params:  dict,
        dx:      float = 1.0,
        dy:      float | None = None,
    ) -> 'Domain':
        if dy is None:
            dy = dx
        nx_in, ny_in, solid, bc_type = _parse_mesh_ascii(map_str)
        return cls(nx_in, ny_in, dx, dy, solid, bc_type, dict(params))

    @classmethod
    def from_text(cls, text: str) -> 'Domain':
        """
        Build a Domain from a plain-text spec.  Two sections separated by a
        line containing only '---':

          # Optional header — '# comment' lines and 'key: value' lines.
          # Recognised keys:
          #   rho, nu, lid_u, inlet_u, inlet_v   → fluid / BC parameters
          #   Lx, Ly                              → total domain lengths
          #   x_axis, y_axis                      → stretching spec
          #                                         ('uniform' | 'tanh beta=…'
          #                                          | 'list w1 w2 …')

          ---

          ##############
          >            <
          >            <
          ##############

        A file with no '---' separator is parsed as a plain ASCII map with
        all defaults (Lx=Ly=1, uniform spacing, no extra parameters).
        """
        if '---' in text:
            header_text, mesh_text = text.split('---', 1)
        else:
            header_text, mesh_text = '', text

        params = {}
        Lx, Ly = 1.0, 1.0
        x_axis = 'uniform'
        y_axis = 'uniform'

        for raw in header_text.splitlines():
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            if ':' not in line:
                raise ValueError(f"header line must be 'key: value': {raw!r}")
            key, val = line.split(':', 1)
            key, val = key.strip(), val.strip()
            if key in ('rho', 'nu', 'lid_u', 'inlet_u', 'inlet_v'):
                params[key] = float(val)
            elif key == 'Lx':     Lx     = float(val)
            elif key == 'Ly':     Ly     = float(val)
            elif key == 'x_axis': x_axis = val
            elif key == 'y_axis': y_axis = val
            else:
                raise ValueError(f"Unknown header key: {key!r}")

        nx, ny, solid, bc_type = _parse_mesh_ascii(mesh_text)
        dx_arr = _parse_axis(x_axis, nx, Lx)
        dy_arr = _parse_axis(y_axis, ny, Ly)
        return cls(nx, ny, dx_arr, dy_arr, solid, bc_type, params)

    @classmethod
    def from_file(cls, path: str) -> 'Domain':
        """Load a Domain from a .cfd text file (see from_text for format)."""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_text(f.read())

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

    @classmethod
    def stretched_closed(
        cls,
        nx: int, ny: int,
        Lx: float = 1.0,
        Ly: float = 1.0,
        beta_x: float = 1.0,
        beta_y: float = 2.5,
        params: dict | None = None,
    ) -> 'Domain':
        """
        Closed lid-driven cavity on a tanh-stretched grid.

        beta_x / beta_y = 1.0 → uniform along that axis.  Larger values
        cluster cells near both walls.  Useful for resolving boundary
        layers at moderate Reynolds numbers without scaling the cost as
        nx² / ny².
        """
        dx_arr = tanh_spacing(nx, Lx, beta_x)
        dy_arr = tanh_spacing(ny, Ly, beta_y)
        bc_type = {
            'left': 'no_slip', 'right': 'no_slip',
            'bottom': 'no_slip', 'top': 'lid',
        }
        return cls(nx, ny, dx_arr, dy_arr,
                   np.zeros((nx, ny), dtype=bool),
                   bc_type, params or {})
