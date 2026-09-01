# NBody-EuroHPC reference audit

Reference inspected: [albtad01/NBody-EuroHPC](https://github.com/albtad01/NBody-EuroHPC), main commit `2f3d66c` (4 March 2026).

## What the blue pulse is

The pulse in `assets/demo.gif` is an authored OpenGL strobe, not simulated
physics. `OGLSpheresVisuGS.cpp`:

1. assumes the track “Move Your Body” is 130 BPM;
2. derives a sharp wall-clock pulse from a sine wave raised to the eighth power;
3. normalises each particle's squared velocity;
4. adds the pulse to particles above a velocity threshold, whitening the fast
   cyan particles on each beat.

The source comments call this the “CYBERPUNK AUDIO-SYNC LOGIC” and “Energy
Core” pulsing effect. It is independent of gravitational density, force, and
time integration. In the visual-demo suite, blue means Milky Way tracer-mass
density; there is no time-based blue strobe.

## Why this repository keeps its own collision

The MUrB repository is useful as a progressive CPU/CUDA performance benchmark,
but its current main branch is not the safer live collision application:

- Its default `galaxy` initial condition is a rotating spherical shell around
  a central body, not a Milky Way–Andromeda encounter.
- Its separate Milky Way/Andromeda loader requires
  `milkyway_andromeda.tab`, which is not present in the repository.
- The README advertises many implementation tags, but `createImplem()` wires
  only `cpu+naive`, `cpu+omp`, `mpi`, `bin+player`, and—under CUDA—
  `gpu+multinode`. The documented single-GPU command uses `gpu+tile+full`,
  which currently falls through to a null simulation pointer.
- Its live renderer needs OpenGL. CINECA currently documents Xorg as disabled
  on Leonardo Booster GPU nodes; a headless batch simulation plus local replay
  avoids that event-day dependency.

The local collision already has real Local Group units and parameters,
headless frames, deterministic replay, an observational-uncertainty sweep, and
one NumPy/CuPy source. It has therefore been retained and hardened with CPU
chunk parallelism, GPU-resident leapfrog updates, reduced device transfers,
and separate DCGP/Booster job templates.

## What has now been adapted as a separate demo

`galaxy_collision_3d` adopts MUrB's most useful computational idea without
turning the restricted visual model into something it is not. It applies a
softened, direct, all-pairs 3-D force to massive Milky Way and M31 disc, bulge,
and halo super-particles. Its GPU force loop uses the reference's shared-memory
tiling strategy; its CPU path evaluates the identical equation in bounded
NumPy tiles. It retains this project's Local Group encounter parameters,
headless output contract, solver selection, and Leonardo CPU/GPU launch path.
The new rotatable viewer consumes computed 3-D positions, not an authored
OpenGL pulse.

See [`GALAXY_COLLISION_3D.md`](GALAXY_COLLISION_3D.md) for the observational
inputs, limitations, and launch details.

## What remains useful from MUrB

MUrB can still be shown as a scaling/optimisation benchmark if its launch
factory, missing data, and SLURM scripts are fixed upstream. That should be a
separate demo story from the scientifically labelled galaxy collision. An
all-pairs direct solver demonstrates kernel optimisation; the restricted
collision demonstrates a plausible encounter with many visual tracers. They
answer different questions and should not be presented as interchangeable.

## The mathematical difference

MUrB's naïve force kernel is a conventional, fully self-gravitating 3D
N-body calculation. Every massive body accelerates every other body using a
softened Newtonian pair force, so one force evaluation costs O(N²):

`a_i = Σ_j G m_j (q_j - q_i) / (|q_j - q_i|² + ε²)^(3/2)`.

This demo is a restricted 2D encounter model. Only two softened Plummer
galaxy centres carry gravitational mass. The visible disc points are massless
tracers: each feels both galaxy potentials, but tracers do not attract each
other and cannot back-react on the centres. The two centres do attract one
another. A force evaluation therefore costs O(N), making tens or hundreds of
thousands of visually useful tracers practical during a live event. It is a
model of tidal morphology, not a replacement for a self-consistent galactic
N-body research calculation.

The frontend now keeps two independent controls:

- **Compute** selects where array work runs: CPU, GPU, or the GPU/CPU encoding
  pipeline.
- **Solver** selects how the state advances in time: second-order symplectic
  leapfrog (the default), first-order symplectic Euler, MUrB's ordinary
  constant-acceleration update, or classical RK4.

MUrB's ordinary update uses
`q_new = q + v Δt + ½ a Δt²; v_new = v + a Δt`. Its separate CUDA leapfrog
implementation uses staggered half-step velocities. The local default is the
standard kick–drift–kick leapfrog/velocity-Verlet form, which recomputes force
after the drift and is the better default for long orbital trajectories.

The inspected commit also advertises Barnes–Hut, but the corresponding
`SimulationNBodyCPUBarnersHut` files currently duplicate the naïve class and
`createImplem()` does not wire that mode. It has therefore not been copied or
presented here as a working O(N log N) implementation.
