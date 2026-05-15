"""
Geometric primitives for building Domain solid masks.

Each shape exposes `inside(X, Y)` (and `inside3d(X, Y, Z)` where it makes
sense) returning a boolean array of the same shape as the inputs.  Domain
factories OR these into the cell solid mask, so shapes can stack
additively on top of any ASCII-encoded geometry.

Adding a new shape: subclass `Shape`, implement `inside`, register it in
`SHAPE_REGISTRY` for the `.cfd` parser to find by name.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Shape:
    """
    Base class.  Subclasses implement `signed_distance(X, Y)` (positive
    outside the body, negative inside) so callers can build either a hard
    mask (`inside`) or a smooth volume fraction `chi ∈ [0, 1]` (`chi_at`).

    Attributes:
      kind     'solid' or 'fluid' — adds or carves out, respectively
      epsilon  smoothing length, in cells.  epsilon=0 gives the original
               binary mask; epsilon>0 returns a tanh-smoothed indicator
               that spans roughly 2·epsilon cells across the interface
               (used by the solver's volume-penalisation step).
    """
    kind:    str   = 'solid'
    epsilon: float = 0.0

    def signed_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def inside(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.signed_distance(X, Y) <= 0.0

    def chi_at(self, X: np.ndarray, Y: np.ndarray, h: float) -> np.ndarray:
        """Volume fraction inside this shape (1 = solid, 0 = fluid)."""
        phi = self.signed_distance(X, Y)
        if self.epsilon <= 0.0:
            return (phi <= 0.0).astype(np.float32)
        scale = max(self.epsilon * h, 1e-12)
        return 0.5 * (1.0 - np.tanh(phi / scale)).astype(np.float32)


@dataclass
class Circle(Shape):
    cx: float = 0.0
    cy: float = 0.0
    r:  float = 0.1

    def signed_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.sqrt((X - self.cx) ** 2 + (Y - self.cy) ** 2) - self.r


@dataclass
class Rectangle(Shape):
    x0: float = 0.0
    x1: float = 0.0
    y0: float = 0.0
    y1: float = 0.0

    def signed_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # Standard 2-D rectangle SDF: positive outside, negative inside.
        dx = np.maximum(self.x0 - X, X - self.x1)
        dy = np.maximum(self.y0 - Y, Y - self.y1)
        outside = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)
        inside  = np.minimum(np.maximum(dx, dy), 0.0)
        return outside + inside


# Names recognised by the `.cfd` parser.  Each entry maps a shape keyword to
# (class, required-keys, optional-keys).
SHAPE_REGISTRY = {
    'circle': (Circle,    ('cx', 'cy', 'r'),         ()),
    'rect':   (Rectangle, ('x0', 'x1', 'y0', 'y1'),  ()),
}


def parse_shape_spec(spec: str) -> Shape:
    """
    Parse one `shape:` value into a Shape.  Examples:
        "circle cx=0.5 cy=0.5 r=0.1"
        "rect x0=0.1 x1=0.3 y0=0.4 y1=0.6 solid"
        "circle cx=0.8 cy=0.5 r=0.15 epsilon=2"   ← smooth wall

    Trailing 'solid' (default) or 'fluid' picks the shape kind.
    Optional `epsilon=N` enables tanh-smoothed walls spanning ~2·N cells.
    """
    parts = spec.strip().split()
    if not parts:
        raise ValueError("empty shape spec")
    name   = parts[0]
    kwargs = {}
    kind   = 'solid'
    epsilon = 0.0
    for tok in parts[1:]:
        if '=' in tok:
            k, v = tok.split('=', 1)
            if k == 'epsilon':
                epsilon = float(v)
            else:
                kwargs[k] = float(v)
        elif tok in ('solid', 'fluid'):
            kind = tok
        else:
            raise ValueError(
                f"shape token must be 'key=value' or 'solid'/'fluid': {tok!r}")
    if name not in SHAPE_REGISTRY:
        raise ValueError(f"unknown shape kind: {name!r} (known: {list(SHAPE_REGISTRY)})")
    cls, required, _optional = SHAPE_REGISTRY[name]
    missing = [k for k in required if k not in kwargs]
    if missing:
        raise ValueError(f"shape {name!r} missing required keys: {missing}")
    return cls(kind=kind, epsilon=epsilon, **kwargs)
