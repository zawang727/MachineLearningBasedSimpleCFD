# MachineLearningBasedSimpleCFD

A 2D incompressible Navier-Stokes CFD solver with a U-Net deep learning surrogate.
Define any geometry and boundary conditions using a plain ASCII map, run the numerical solver,
then train a CNN to predict the same steady-state flow field in milliseconds.

## Physics

Incompressible 2D Navier-Stokes (projection / Chorin method):

```
∂u/∂t + u·∇u = −(1/ρ)∇p + ν∇²u
∇·u = 0
```

Discretised on a **MAC staggered grid**:
- Pressure `p` at cell centres `(nx, ny)`
- x-velocity `u` at right faces `(nx+1, ny)`
- y-velocity `v` at top faces `(nx, ny+1)`

Stability: `dt ≤ min(dx/U, 0.5·dx²/ν)` — computed automatically.

## `.cfd` Input File

Bundles fluid properties, axis stretching, and the cell map in a single
plain-text file.  Faster to debug than embedded Python strings — edit, save,
re-run.

```
# cases/lid_driven_cavity.cfd
rho:    1.0
nu:     0.01            # Re = U_lid * L / nu = 100
lid_u:  1.0

Lx:     1.0
Ly:     1.0
x_axis: uniform
y_axis: tanh beta=2.5   # cluster cells near top and bottom walls

---
------------------
#                #
#                #
... (16 fluid rows)
#                #
##################
```

Run it through the generic driver:

```bash
python cases/run.py cases/lid_driven_cavity.cfd
python cases/run.py cases/channel_flow.cfd --duration 15
```

Header keys (all optional; everything has a default):

| Key | Meaning |
|---|---|
| `rho`, `nu` | Fluid density and kinematic viscosity |
| `lid_u`, `inlet_u`, `inlet_v` | BC magnitudes |
| `Lx`, `Ly` | Total domain lengths (default 1.0 each) |
| `x_axis`, `y_axis` | Cell-width stretching spec (see below) |

Stretching specs:
- `uniform` — equal-width cells.
- `tanh beta=2.5` — double-sided tanh clustering; larger β = tighter wall layers.
- `list 0.02 0.03 0.04 ...` — explicit cell widths (must sum to `Lx`/`Ly`).

**Geometric primitives** can be added with one or more `shape:` lines in
the header.  Cells whose centres land inside the shape become solid (or
fluid, if the trailing keyword is `fluid`).  Stacks additively on top of
any ASCII-encoded geometry.

```
shape: circle cx=0.8 cy=0.5 r=0.15
shape: rect   x0=1.5 x1=1.8 y0=0.3 y1=0.7 solid
```

See `cases/cylinder_flow.cfd` for a flow-past-cylinder example.  This is
the Phase-2 entry point: walls are stair-stepped (first-order in space).
True cut-cell or smooth penalization for curved walls is the next
Phase-2 commit.

A file with no `---` separator is parsed as a plain ASCII map with all
defaults — preserves the legacy embedded-string layout.

**3-D `.cfd` files** add `nz`, `Lz`, and `z_axis` to the header.  The 2-D
ASCII map after `---` is extruded `nz` times in z; front and back walls
default to no-slip.  The presence of `nz:` in the header tells
`cases/run.py` to dispatch through `Domain3D` / `Solver3D` instead of the
2-D path.

```
# cases/lid_driven_cavity_3d.cfd
rho: 1.0
nu:  0.01
lid_u: 1.0
Lx: 1.0  Ly: 1.0  Lz: 1.0
nz: 16
y_axis: tanh beta=2.5
z_axis: tanh beta=1.5
---
<2-D map>
```

## ASCII Map Interface

Define geometry by writing a text map. Each character sets a cell or face type:

| Symbol | Meaning |
|--------|---------|
| `#`    | No-slip wall (u = v = 0) |
| `-`    | Moving horizontal lid (+x at `params['lid_u']`) |
| `>`    | Inlet left face (u = `params['inlet_u']`, v = 0) |
| `<`    | Outlet right face (zero-gradient, p = 0) |
| `^`    | Inlet bottom face (v = `params['inlet_v']`) |
| `*`    | Solid obstacle block (no-slip) |
| ` ` `.`| Fluid cell |

Top row = high y. Example — flow around a block:

```
#############################
>                            <
>       ****                 <
>       ****                 <
>       ****                 <
>                            <
#############################
```

```python
from cfd import Domain, Solver, Material

domain   = Domain.from_ascii(ascii_map, params={'inlet_u': 1.0, 'rho': 1.0, 'nu': 0.01},
                              dx=1/40, dy=1/20)
material = Material(rho=1.0, nu=0.01)
solver   = Solver(domain, material)       # dt auto-computed
state    = solver.run(duration=20.0)
state.plot(save_path="results/flow.png")
```

## Project Structure

```
MachineLearningBasedSimpleCFD/
├── cfd/
│   ├── domain.py          # ASCII → Domain
│   ├── solver.py          # Projection method NS solver
│   ├── material.py        # Fluid properties (ρ, ν, Re)
│   └── visualization.py   # Velocity, pressure, streamlines, comparison
├── models/
│   └── unet.py            # 2D U-Net surrogate (PyTorch, ~55K params)
├── cases/
│   ├── lid_driven_cavity.py
│   ├── channel_flow.py
│   └── flow_around_block.py
├── generate_data.py
├── train.py
└── predict.py
```

## Test Cases

### 1. Lid-Driven Cavity

```
-              -    ← top wall moves at U=1 (+x)
#              #
#              #
-              -    ← bottom wall (optional, set to no-slip for classic case)
```

```bash
python cases/lid_driven_cavity.py --Re 400 --nx 64 --ny 64
python cases/lid_driven_cavity.py --all-Re
```

Comparison against Ghia (1982) centreline profiles at Re=100, 400, 1000.

### 2. Channel Flow (Poiseuille)

```
################
>              <
>              <
################
```

```bash
python cases/channel_flow.py --Re 100
```

Validates against analytical: `u(y) = 6·U_avg·y·(H−y)/H²`

### 3. Flow Around a Square Block

```bash
python cases/flow_around_block.py --Re 100 --block-w 4 --block-h 4
```

Recirculation wake downstream of block. Vortex shedding at higher Re.

## ML Surrogate

### Generate dataset

```bash
python generate_data.py --n-per-case 3 --output data.npz
```

Each sample: (obstacle mask, inlet BC map, lid BC map, dx/Lx map, dy/Ly map)
→ (u, v, p) steady-state fields.  The two trailing channels make the
surrogate mesh-aware so the same network can train on uniform and
stretched grids; in 3D the input is 6-channel (adds dz/Lz).

### Train

```bash
python train.py --data data.npz --epochs 100 --model-out model.pt
```

### Predict and compare

```bash
python predict.py --model model.pt --data data.npz --n-samples 3
```

Saves `results/sample_XX_comparison.png` — 3×2 grid (rows = u/v/p, cols = NN/CFD).

## Model Architecture

Shallow U-Net with skip connections:

```
Input (3, ny, nx)
  → Enc: Conv(3→16) + Conv(16→32) + Conv(32→64)  [with MaxPool]
  → Bot: Conv(64→128)
  → Dec: Upsample + skip connections
  → Out: Conv(16→3, 1×1)
≈ 55,000 parameters
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. No GPU required (solver and training both work on CPU).

## Roadmap

- [ ] Phase 2: Time-dependent prediction, animated vortex shedding
- [ ] Phase 3: 3D extension (3D lid-driven cavity, 3D channel)
- [ ] Phase 4: Physics-informed loss (divergence-free constraint)
- [ ] Phase 5: Interactive web demo

## Meshing Roadmap

Today the solver lives on a uniform Cartesian MAC grid: pressure at cell
centres, velocities at faces, scalar `dx, dy, dz`. That covers all current
test cases (cavity, channel, square block) but excludes curved walls,
near-wall refinement, and CAD-style geometry. The plan below grows the
mesh capability in four stages of increasing scope. Each stage builds on
the previous one; the CNN surrogate survives stages 1–2 but breaks at
stage 3.

### Stage 1 — Non-uniform structured grid
Keep the `(nx, ny, nz)` MAC layout, but replace scalar spacings with per-axis
arrays `dx[i], dy[j], dz[k]`. Stencils become spacing-aware; the pressure
Laplacian stays tri-diagonal per axis with variable coefficients. Enables
near-wall clustering (boundary layers) and far-field stretching.

- **ASCII format**: add a header that declares stretching, e.g.
  ```
  axes: x=uniform(0,1,nx=64) y=tanh(0,1,ny=48,ratio=4)
  ```
  followed by the existing cell map.
- **Solver changes**: `domain.dx → domain.dx[:]`; refactor stencils in
  `solver.py` / `solver3d.py` to use local `dx[i]` / `dy[j]` / `dz[k]`.
- **ML pipeline**: still works — CNN sees regular `(nz, ny, nx)` tensors;
  add `dx/dy/dz` arrays as extra input channels or metadata.
- **Cost**: ~1–2 weeks. Existing tests still apply.

### Stage 2 — Cut-cell / immersed boundary on Cartesian
Each cell carries a fluid volume fraction `α_v ∈ [0,1]` and per-face open-area
fractions. Curved geometry is represented by a signed-distance field or a
boundary polyline; the underlying grid stays uniform Cartesian. This is the
approach used by AMR codes (Basilisk, AMReX) for airfoils, cylinders, hulls.

- **ASCII format**: extend with geometric primitives or an SDF, e.g.
  ```
  shape: circle cx=0.5 cy=0.5 r=0.2 solid
  shape: polyline (0.1,0.4)(0.3,0.5)(0.5,0.4) wall
  ```
  Or accept a separate `.sdf` / image file as the boundary.
- **Solver changes**: face fluxes scaled by open-area fraction; wall flux
  via a cut-face boundary term. Small-cell timestep restrictions need
  mitigation (cell-merging or flux redistribution).
- **ML pipeline**: still works — `α_v` and the SDF become additional CNN
  input channels.
- **Cost**: ~3–4 weeks. Adds new test cases (flow over cylinder, NACA).

### Stage 3 — Unstructured finite-volume (hex / tetra / mixed)
The architectural break. Replace `(nx, ny, nz)` arrays with cell, face, and
node tables. Fluxes are computed face-by-face. Pressure projection becomes
a general sparse system (CG + algebraic multigrid, not LU).

- **Mesh format**: adopt Gmsh `.msh` or VTK legacy `.vtk` rather than
  inventing a format. Optionally keep a thin ASCII generator for simple
  built-in cases (channels, ducts) that emits `.msh` under the hood.
- **Solver changes**: full rewrite — MAC layout no longer applies. Choose
  one of:
  - Collocated FV with Rhie–Chow interpolation (simpler), or
  - Face-staggered unstructured (closer in spirit to current MAC).
- **ML pipeline**: CNN dies. Options:
  - Graph neural network on the mesh (MeshGraphNets style).
  - Interpolate fields onto a background Cartesian grid for the surrogate
    only — keeps the U-Net usable as a coarse predictor.
- **Cost**: ~2–3 months. Effectively a new codebase sharing the same repo.

### Stage 4 — Mesh adaptivity (AMR)
Refinement criteria drive on-the-fly mesh changes. Only worth doing after
stage 3 is solid; relies on the same unstructured data structures plus
refine/coarsen operators and load balancing.

### Where the cliff is
Stage 3 ends the CNN surrogate path. If the ML pipeline is core to the
project's value, stage 2 (cut-cell on Cartesian) covers most "curved
geometry" use cases — cylinders, airfoils, organic shapes — while keeping
the U-Net training feasible. Pick stage 3 only if you need true arbitrary
geometry (CAD imports, complex internal passages).

### Suggested entry point
Stage 1. It is small enough to validate against the existing cavity /
channel / block cases (refining near the lid should converge faster and
match Ghia at lower `nx`). After that, the choice between stage 2 and
stage 3 follows from which geometries you actually want to simulate.
