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
    """Base class — every shape has a `kind` ('solid' for now)."""
    kind: str = 'solid'

    def inside(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Circle(Shape):
    cx: float = 0.0
    cy: float = 0.0
    r:  float = 0.1

    def inside(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (X - self.cx) ** 2 + (Y - self.cy) ** 2 <= self.r ** 2


@dataclass
class Rectangle(Shape):
    x0: float = 0.0
    x1: float = 0.0
    y0: float = 0.0
    y1: float = 0.0

    def inside(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return ((self.x0 <= X) & (X <= self.x1)
                & (self.y0 <= Y) & (Y <= self.y1))


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

    The trailing keyword 'solid' (default) or 'fluid' picks the shape kind.
    """
    parts = spec.strip().split()
    if not parts:
        raise ValueError("empty shape spec")
    name   = parts[0]
    kwargs = {}
    kind   = 'solid'
    for tok in parts[1:]:
        if '=' in tok:
            k, v = tok.split('=', 1)
            kwargs[k] = float(v)
        elif tok in ('solid', 'fluid'):
            kind = tok
        else:
            raise ValueError(f"shape token must be 'key=value' or 'solid'/'fluid': {tok!r}")
    if name not in SHAPE_REGISTRY:
        raise ValueError(f"unknown shape kind: {name!r} (known: {list(SHAPE_REGISTRY)})")
    cls, required, _optional = SHAPE_REGISTRY[name]
    missing = [k for k in required if k not in kwargs]
    if missing:
        raise ValueError(f"shape {name!r} missing required keys: {missing}")
    return cls(kind=kind, **kwargs)
