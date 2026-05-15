from .domain import Domain
from .domain3d import Domain3D
from .material import Material
from .solver import Solver, FlowState
from .solver3d import Solver3D, FlowState3D
from .shapes import Shape, Circle, Rectangle, parse_shape_spec
from .visualization import plot_velocity, plot_pressure, plot_fields, plot_comparison
from .visualization3d import (plot_velocity_3d, plot_fields_3d, plot_comparison_3d,
                              plot_3d_slices, plot_3d_vectors, plot_3d_case_summary)
