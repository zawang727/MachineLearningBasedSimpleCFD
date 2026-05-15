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

Each sample: (obstacle mask, inlet BC map, lid BC map) → (u, v, p) steady-state fields.

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
