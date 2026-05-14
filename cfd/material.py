from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Material:
    """Incompressible Newtonian fluid properties."""
    rho: float = 1.0    # density  [kg/m³]
    nu:  float = 0.01   # kinematic viscosity  [m²/s]

    @property
    def mu(self) -> float:
        return self.rho * self.nu

    def reynolds(self, U: float, L: float) -> float:
        return U * L / self.nu

    @classmethod
    def from_Re(cls, Re: float, U: float = 1.0, L: float = 1.0, rho: float = 1.0) -> Material:
        return cls(rho=rho, nu=U * L / Re)
