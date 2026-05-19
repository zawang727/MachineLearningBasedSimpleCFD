# Stage 3 — Unstructured Finite Volume

Detailed multi-session plan for the architectural break from Cartesian MAC
to general unstructured FV.  Each phase is a stand-alone commit-sized
piece; the phases stack so you can stop after any of them and still have
a working improvement.

## Where we are

- **Phase 3a** ✅ landed: mesh data structures (`UnstructuredMesh2D`),
  `rect_triangulation` generator, first-order FV Poisson assembly
  (`assemble_poisson` / `solve_poisson`).  Single-pair stencil — bottoms
  out at ~5 % L² error on non-orthogonal triangle meshes.

- **Phase 3b** ✅ landed: Green-Gauss cell gradient
  (`green_gauss_gradient`) + over-relaxed non-orthogonal correction
  applied as a deferred RHS update inside an outer fixed-point loop
  (`solve_poisson_2nd_order`).  Convergence rate on the sin·sin test
  case is now 1.91, 1.96, 1.98 over n ∈ {32, 64, 128} — **second-order
  accuracy restored**.  At n=128 the L² error drops to 2.1e-5, ~2600×
  smaller than the first-order plateau.

- **Phase 3c–3h** ahead.

## Why this is multi-session

The README estimate was 2–3 months; that's roughly accurate.  Each phase
below introduces real numerical work that needs validation before the
next one is meaningful.  Doing it well > doing it fast.

---

## Phase 3b — Second-order FV (non-orthogonal correction)

**Goal**: drop the L² error below 1% on the existing rect-triangulation
and show second-order convergence (slope ≈ 2 on a log-log refinement
plot).

**Work**:
1. Add `cell_gradient(p)` reconstruction:
   - Green-Gauss formula: ∇p_c ≈ (1/V_c) Σ_f p_f · A_f · n̂_f, where p_f
     is the face-averaged value.
   - On first pass, take p_f = mean(p_L, p_R); later iterate with the
     non-orthogonal correction loop.
2. Re-derive the face flux as
   $\text{flux}_f = (p_R - p_L) / d_⟂ + (∇p_f \cdot t̂) · (\text{tangential
   offset of cell-to-cell line})$,
   where ∇p_f is interpolated from the two cell gradients.
3. Iterate: 2–3 deferred-correction sweeps per linear solve.
4. Validate with the existing Poisson demo; verify slope ≈ 2 on
   `rect_triangulation(n)` for n = 16, 32, 64, 128.
5. Add a *non-orthogonal* test: random Delaunay triangulation of the unit
   square.  Slope should still be ≈ 2.

**Risk**: iterative gradient reconstruction is tricky to get
matrix-symmetric.  May need to fall back to explicit deferred correction
(non-symmetric L, GMRES instead of CG).

**Effort**: 3–5 days of focused work.

---

## Phase 3c — Generalise the mesh: quads and mixed polyhedra

**Goal**: replace the fixed-size `(n_cells, 3)` cells array with a flexible
representation that supports triangles, quadrilaterals, and mixed
elements; same code path works for both.

**Work**:
1. Refactor `cells` to a flat connectivity (`cell_node_offsets` +
   `cell_nodes`), VTK-style.  Update `_build_faces` to handle variable
   nodes-per-cell.
2. Re-test the existing triangle mesh.
3. Add `quad_mesh(nx, ny)` generator → orthogonal Cartesian quads.
   Validate Poisson on quads: should match the structured solver to
   round-off (this becomes the FV cross-check).
4. Add a hybrid generator that mixes triangles and quads.

**Effort**: 2–3 days.

---

## Phase 3d — Mesh I/O: read Gmsh `.msh`

**Goal**: stop using only synthetic meshes — accept external geometry.

**Work**:
1. Implement a minimal Gmsh v4 / v2 ASCII reader (just nodes + elements;
   ignore physical groups for now).
2. Hello-world: read a 2-D mesh of a cylinder-in-channel produced by
   `gmsh` and solve Poisson on it.  Compare to the existing
   `cylinder_flow.cfd` Cartesian result.
3. Optional: write meshio as a dependency so we get Gmsh + VTK + STL for
   free.  Cleaner long-term, adds a dep.

**Effort**: 2 days for a minimal reader, ~1 day with meshio.

---

## Phase 3e — Steady Stokes on unstructured mesh

**Goal**: first momentum-equation solve on the unstructured grid.

**Pick a layout**:
- **Collocated** (u, v, p at cell centres) with Rhie–Chow interpolation
  to prevent checkerboarding — simpler implementation, the standard in
  modern finite-volume codes.
- *Or* face-staggered (u·n̂ on faces, p at cells) — closer in spirit to
  the existing MAC solver but harder to assemble.

Default to collocated + Rhie–Chow.

**Work**:
1. Implement steady-state Stokes: ν ∇²u − ∇p = f, ∇·u = 0.  No
   convection, no time-stepping.
2. SIMPLE iteration: solve momentum predictor, then pressure correction
   from the discrete continuity equation, update u, repeat to
   convergence.
3. Apply Phase 3b's gradient machinery to the diffusion term.
4. Validate against the analytical 1-D Poiseuille profile on a 2-D
   channel mesh.
5. Add the Stokes equivalent of `cylinder_flow.cfd` and compare to the
   Phase 2 penalisation result.

**Effort**: 1–2 weeks.

---

## Phase 3f — Unsteady incompressible Navier–Stokes

**Goal**: full transient NS solver on unstructured mesh.  Functional
replacement for the existing `Solver` / `Solver3D`.

**Work**:
1. Add convection term: face flux $\rho u u \cdot n̂$ with upwinded
   density-weighted average.
2. Time-stepping: implicit/explicit Euler or Crank–Nicolson for
   diffusion; explicit upwind for convection (CFL-bound dt).
3. Projection or SIMPLE/PISO for pressure–velocity coupling.
4. Boundary conditions: no-slip / inlet / outlet on tagged face sets
   (extends the current `bc_type` dict from 4 walls to N tagged groups).
5. Validate against existing Phase 2 cases (cylinder at Re=40, lid-driven
   cavity).

**Effort**: 3–4 weeks.

---

## Phase 3g — 3-D extension

Same as 3e–3f but in 3-D: tetrahedral and hexahedral cells, 3-D face
geometry, sphere obstacles.  Most of the framework code is dimension-
agnostic if Phase 3c was done right.

**Effort**: 2–3 weeks on top of 3f.

---

## Phase 3h — ML pipeline: graph neural network

The CNN dies at the unstructured break.  Two paths:

### Path A — GNN surrogate (the proper answer)
1. Adopt PyTorch Geometric or DGL (heavy dependency).
2. Encode each sample as a graph: nodes = cells, edges = shared faces,
   features = (χ, BC encoding, cell volume, face-normal vector,
   neighbour-distance) at each node.
3. Implement a MeshGraphNets-style message-passing network.
4. Re-train and validate against Stage 3 CFD outputs.

**Effort**: 2–3 weeks.

### Path B — Background grid surrogate (cheaper interim)
1. Interpolate the unstructured CFD output onto a fixed Cartesian
   background grid.
2. Train the existing U-Net on that.
3. At inference, interpolate the prediction back onto the unstructured
   mesh.

**Effort**: 3–5 days.  Loses spatial detail near small features.

---

## Cross-cutting items

These run alongside the phases above:

- **Solver scalability**: 100k+ cell meshes need a Krylov solver (CG for
  symmetric Poisson, GMRES for asymmetric).  Pre-condition with smoothed
  aggregation AMG (PyAMG).  LU stops scaling past ~50k unknowns.
- **Visualisation**: tripcolor / matplotlib works at small meshes but
  becomes slow > 50k cells.  ParaView via VTK output (write `.vtu`
  files) is the right answer.
- **Test harness**: each phase needs a regression test that runs in
  seconds.  Build a small `tests/` directory now to keep refactors
  honest.

---

## What to do *next*

Phase 3b (non-orthogonal correction) is the highest-leverage next step:
it turns the prototype into a real second-order FV code, which is the
load-bearing prerequisite for everything from 3e onward.  Anything more
ambitious without 3b is building on first-order foundations.

Approximate cumulative effort from here:
- Phase 3b alone: 1 week → second-order Poisson on triangles.
- Through 3e: 1 month → first NS prototype on unstructured.
- Through 3g: 2–3 months → full 3-D unstructured NS replacing the
  current Cartesian solver.
- Through 3h: another 2–3 weeks → ML pipeline restored.
