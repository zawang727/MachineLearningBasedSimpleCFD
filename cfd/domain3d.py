from __future__ import annotations
import numpy as np
from .domain import Domain as _Domain2D
from .domain import (_as_spacing, _scalar_summary,
                     _parse_axis, _parse_mesh_ascii, tanh_spacing)


class Domain3D:
    """
    3-D rectangular domain for MAC-grid NS solver.

    Grid: nx × ny × nz cells.  All cells are real fluid; walls live on the
    bounding u / v / w faces.
      p[i,j,k]   pressure at cell centre         (nx, ny, nz)
      u[i,j,k]   x-velocity at right x-face      (nx+1, ny, nz)
      v[i,j,k]   y-velocity at top  y-face       (nx, ny+1, nz)
      w[i,j,k]   z-velocity at front z-face      (nx, ny, nz+1)

    Spacing: dx / dy / dz may be scalars (uniform mesh) or 1-D arrays of
    shape (nx,) / (ny,) / (nz,) for stretched meshes.  Per-axis spacing
    arrays (`dx_arr`, `dy_arr`, `dz_arr`) and cell / face coordinate arrays
    (`x_cell`, `y_cell`, `z_cell`, `x_face`, `y_face`, `z_face`) are always
    exposed so downstream code does not have to special-case uniformity.

    ASCII formats
    -------------
    from_ascii(map_str, params, dx, nz)
        Extrude a single 2D cross-section uniformly in z.
        z-faces default to no_slip.

    from_ascii_layers(layers_str, params, dx)
        Parse nz 2D ASCII maps separated by '===' lines.
        Each map defines one z-layer (k=0 = first map in string).
        Same symbol table as Domain.from_ascii: '#' no-slip, '-' lid,
        '>' inlet, '<' outlet, '*' solid, ' '/'.' fluid.
    """

    def __init__(
        self,
        nx: int, ny: int, nz: int,
        dx: 'float | np.ndarray',
        dy: 'float | np.ndarray',
        dz: 'float | np.ndarray',
        solid: np.ndarray,     # (nx, ny, nz) bool
        bc_type: dict,         # 'left'/'right'/'bottom'/'top'/'front'/'back'
        bc_values: dict,
    ) -> None:
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx_arr = _as_spacing(dx, nx, 'dx')
        self.dy_arr = _as_spacing(dy, ny, 'dy')
        self.dz_arr = _as_spacing(dz, nz, 'dz')
        self.dx     = _scalar_summary(self.dx_arr)
        self.dy     = _scalar_summary(self.dy_arr)
        self.dz     = _scalar_summary(self.dz_arr)
        self.solid     = solid.astype(bool)
        self.bc_type   = bc_type
        self.bc_values = bc_values

        # ML input encoding maps (nx, ny, nz)
        self.inlet_u_map = np.zeros((nx, ny, nz), dtype=np.float32)
        self.lid_u_map   = np.zeros((nx, ny, nz), dtype=np.float32)
        u_in  = float(bc_values.get('inlet_u', 0.0))
        lid_u = float(bc_values.get('lid_u',   0.0))
        if bc_type.get('left')   == 'inlet': self.inlet_u_map[0,  :, :] = u_in
        if bc_type.get('top')    == 'lid':   self.lid_u_map[:,  -1, :] = lid_u
        if bc_type.get('bottom') == 'lid':   self.lid_u_map[:,   0, :] = lid_u

    # ------------------------------------------------------------------
    # Mesh geometry
    @property
    def x_face(self) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(self.dx_arr)))

    @property
    def y_face(self) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(self.dy_arr)))

    @property
    def z_face(self) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(self.dz_arr)))

    @property
    def x_cell(self) -> np.ndarray:
        xf = self.x_face
        return 0.5 * (xf[:-1] + xf[1:])

    @property
    def y_cell(self) -> np.ndarray:
        yf = self.y_face
        return 0.5 * (yf[:-1] + yf[1:])

    @property
    def z_cell(self) -> np.ndarray:
        zf = self.z_face
        return 0.5 * (zf[:-1] + zf[1:])

    @property
    def Lx(self) -> float:
        return float(self.dx_arr.sum())

    @property
    def Ly(self) -> float:
        return float(self.dy_arr.sum())

    @property
    def Lz(self) -> float:
        return float(self.dz_arr.sum())

    # ------------------------------------------------------------------
    @classmethod
    def from_ascii(
        cls,
        map_str: str,
        params:  dict,
        dx:      float = 1.0,
        dy:      float | None = None,
        dz:      float | None = None,
        nz:      int   = 16,
    ) -> 'Domain3D':
        """
        Extrude a 2D ASCII cross-section uniformly in z.

        The 2D map uses the same symbol table as Domain.from_ascii:
          '#'  no-slip wall      '-'  moving lid
          '>'  inlet (left)      '<'  outlet (right)
          '*'  solid obstacle    ' '/'.'  fluid cell

        Front (z=0) and back (z=D) faces default to no_slip.
        """
        d2 = _Domain2D.from_ascii(map_str, params, dx=dx, dy=dy or dx)
        solid3d = np.repeat(d2.solid[:, :, np.newaxis], nz, axis=2)
        bc_type = dict(d2.bc_type)
        bc_type.setdefault('front', 'no_slip')
        bc_type.setdefault('back',  'no_slip')
        return cls(d2.nx, d2.ny, nz, d2.dx, d2.dy, dz or dx,
                   solid3d, bc_type, dict(params))

    @classmethod
    def from_ascii_layers(
        cls,
        layers_str: str,
        params:     dict,
        dx:         float = 1.0,
        dy:         float | None = None,
        dz:         float | None = None,
    ) -> 'Domain3D':
        """
        Parse nz 2D ASCII layers separated by '===' lines.

        Each layer uses the 2D symbol table. The first layer in the string
        corresponds to k=0 (front face / low z). All layers must have the
        same nx and ny; the x/y BC type is taken from the first layer.
        """
        layer_strs = [s.strip() for s in layers_str.split('===') if s.strip()]
        if not layer_strs:
            raise ValueError("No layers found in ASCII map string")
        ds = [_Domain2D.from_ascii(ls, params, dx=dx, dy=dy or dx)
              for ls in layer_strs]
        solid3d = np.stack([d.solid for d in ds], axis=2)  # (nx, ny, nz)
        bc_type = dict(ds[0].bc_type)
        bc_type.setdefault('front', 'no_slip')
        bc_type.setdefault('back',  'no_slip')
        return cls(ds[0].nx, ds[0].ny, len(ds), ds[0].dx, ds[0].dy, dz or dx,
                   solid3d, bc_type, dict(params))

    @classmethod
    def closed(
        cls,
        nx: int, ny: int, nz: int,
        dx: float = 1.0,
        dy: float | None = None,
        dz: float | None = None,
        params: dict | None = None,
    ) -> 'Domain3D':
        """All-wall closed box. Top face (y=ny-1) is a moving lid (+x)."""
        bc_type = {
            'left':   'no_slip', 'right':  'no_slip',
            'bottom': 'no_slip', 'top':    'lid',
            'front':  'no_slip', 'back':   'no_slip',
        }
        return cls(nx, ny, nz, dx, dy or dx, dz or dx,
                   np.zeros((nx, ny, nz), dtype=bool),
                   bc_type, params or {})

    @classmethod
    def channel(
        cls,
        nx: int, ny: int, nz: int,
        dx: float = 1.0,
        dy: float | None = None,
        dz: float | None = None,
        params: dict | None = None,
    ) -> 'Domain3D':
        """Inlet (+x) on left, outlet on right; all other faces no-slip."""
        bc_type = {
            'left':   'inlet',   'right':  'outlet',
            'bottom': 'no_slip', 'top':    'no_slip',
            'front':  'no_slip', 'back':   'no_slip',
        }
        return cls(nx, ny, nz, dx, dy or dx, dz or dx,
                   np.zeros((nx, ny, nz), dtype=bool),
                   bc_type, params or {})

    @classmethod
    def stretched_closed(
        cls,
        nx: int, ny: int, nz: int,
        Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0,
        beta_x: float = 1.0, beta_y: float = 2.5, beta_z: float = 1.0,
        params: dict | None = None,
    ) -> 'Domain3D':
        """3-D lid-driven cavity on a tanh-stretched grid (clusters near walls)."""
        dx_arr = tanh_spacing(nx, Lx, beta_x)
        dy_arr = tanh_spacing(ny, Ly, beta_y)
        dz_arr = tanh_spacing(nz, Lz, beta_z)
        bc_type = {
            'left':   'no_slip', 'right':  'no_slip',
            'bottom': 'no_slip', 'top':    'lid',
            'front':  'no_slip', 'back':   'no_slip',
        }
        return cls(nx, ny, nz, dx_arr, dy_arr, dz_arr,
                   np.zeros((nx, ny, nz), dtype=bool),
                   bc_type, params or {})

    @classmethod
    def from_text(cls, text: str) -> 'Domain3D':
        """
        Build a Domain3D from a plain-text spec.  Same format as Domain.from_text
        but with z-axis extras.  The header MUST set `nz` to distinguish a 3D
        file from a 2D file; the 2D ASCII map after `---` is extruded `nz`
        times along z.

        Header keys (all optional except `nz` for 3D):
          rho, nu, lid_u, inlet_u, inlet_v     fluid / BC parameters
          Lx, Ly, Lz                            domain lengths (default 1)
          nz                                    # of z-extrusion layers (required)
          x_axis, y_axis, z_axis                stretching specs
        """
        if '---' not in text:
            raise ValueError("3-D input file needs a '---' separator")
        header_text, mesh_text = text.split('---', 1)

        params = {}
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        x_axis = 'uniform'
        y_axis = 'uniform'
        z_axis = 'uniform'
        nz     = None

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
            elif key == 'Lz':     Lz     = float(val)
            elif key == 'nz':     nz     = int(val)
            elif key == 'x_axis': x_axis = val
            elif key == 'y_axis': y_axis = val
            elif key == 'z_axis': z_axis = val
            else:
                raise ValueError(f"Unknown header key: {key!r}")

        if nz is None:
            raise ValueError("3-D input file must declare 'nz: <int>' in the header")

        nx, ny, solid2d, bc_type = _parse_mesh_ascii(mesh_text)
        dx_arr = _parse_axis(x_axis, nx, Lx)
        dy_arr = _parse_axis(y_axis, ny, Ly)
        dz_arr = _parse_axis(z_axis, nz, Lz)

        # Extrude the 2-D solid mask uniformly in z, then default the front /
        # back walls to no-slip (they are not encoded in a 2-D ASCII map).
        solid3d = np.repeat(solid2d[:, :, None], nz, axis=2)
        bc_type = dict(bc_type)
        bc_type.setdefault('front', 'no_slip')
        bc_type.setdefault('back',  'no_slip')

        return cls(nx, ny, nz, dx_arr, dy_arr, dz_arr,
                   solid3d, bc_type, params)

    @classmethod
    def from_file(cls, path: str) -> 'Domain3D':
        """Load a Domain3D from a .cfd text file (see from_text for format)."""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_text(f.read())
