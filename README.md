# Leonardo Visual Demos

A portable exhibition toolkit for **thirteen visually driven HPC demonstrations** designed to run in three modes:

1. **Local development** on an ordinary desktop/laptop.
2. **Desktop GPU** mode using CUDA through CuPy or PyTorch when available.
3. **Leonardo** batch mode, where compute happens headlessly on CINECA's Leonardo system and the results are rendered or replayed on a separate presentation computer.

The project deliberately separates **scientific computation** from **presentation**. Nothing in the viewer requires an X server, OpenGL context, or graphical desktop on the compute node. A simulation writes clean numbered JPEG frames plus small per-frame JSON readouts; the local web viewer streams those files and builds optional titles, legends, controls, and instrumentation as real HTML overlays. Saved runs use the same contract for replay.

## The thirteen demos

Per-demo architecture, scientific limits, extension points, and maintenance
guidance are indexed in [`docs/demos/`](docs/demos/README.md). Shared source
rules live in [`leonardo_demos/demos/AGENTS.md`](leonardo_demos/demos/AGENTS.md).

| ID | Demo | Core computation | Exhibition reveal |
|---|---|---|---|
| `black_hole` | Black-hole lensing | 2-D image-space lens mapping + numerically integrated 3-D photon-path view | Switch between the observer image and the spatial ray-tracing model |
| `pbh` | Primordial black-hole threshold | Batched radial collapse/dispersion demonstrator + adapter for research solver | One universe pulls back into an ensemble and a collapse boundary |
| `fluid` | Virtual wind tunnel | D2Q9 lattice-Boltzmann with advected streaklines | Smooth flow pulls back into millions of updated cells / GPU domains |
| `cosmic_web` | Cosmic-web formation | Particle-mesh gravity, Zel'dovich ICs, H/He gas composition | One universe pulls back into a hydrogen-to-helium composition sweep |
| `galaxy_collision` | Galaxy collision | Restricted N-body, real Milky Way / Andromeda parameters | One future pulls back into the measurement uncertainty that decides it |
| `galaxy_collision_3d` | Galaxy collision - full 3D gravity | Softened direct O(N²) gravity over massive disc, bulge, and halo super-particles | Rotate Gaia/PHAT-informed galaxies freely in space |
| `reaction_diffusion` | Living mathematics | Gray-Scott reaction-diffusion PDE | One pattern pulls back into a parameter-space wall |
| `crystal` | Crystal / snowflake growth | Recursive geometric growth, six habits, deep-zoom tile pyramid | One crystal pulls back into many habits — then zoom into its own branches |
| `neural_wall` | Neural-network wall | Real batched coordinate-network training (PyTorch if available) | "One network" pulls back into many networks trained in parallel |
| `fusion_plasma` | Star in a Bottle | Reduced nonlinear plasma-wave lattice on a tokamak torus | One plasma pulls back into a reactor operating map |
| `plasma_guardian` | AI Plasma Guardian | Batched, differentiable reduced control environment with a trainable PyTorch policy | One unstable shot becomes many virtual training shots |
| `weather_ensemble` | Storm Factory | Barotropic-vorticity atmosphere with advected moisture | One forecast pulls back into many possible storm futures |
| `molecular_dynamics` | Molecular Machine | Coarse-grained 3-D molecular dynamics with all-pairs forces | One trajectory pulls back into a virtual molecular laboratory |

## Quick start (Windows, macOS, Linux)

You need Python **3.10+**.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the address printed in the terminal (normally `http://127.0.0.1:8000`).

On Windows you can also double-click:

```text
scripts/start_viewer_windows.bat
```

For a simple terminal menu that lets you choose a demo, profile, and frame
count, double-click:

```text
Run_Leonardo_Demos.bat
```

On macOS/Linux:

```bash
bash scripts/start_viewer.sh
```

## Run a demo individually

```bash
python run_demo.py reaction_diffusion --profile local --frames 90
python run_demo.py black_hole --profile local --frames 80 --open
python run_demo.py neural_wall --profile local --frames 70
python run_demo.py fusion_plasma --profile local --frames 70
python run_demo.py plasma_guardian --profile local --frames 70
python run_demo.py weather_ensemble --profile local --frames 70
python run_demo.py molecular_dynamics --profile local --frames 70
python run_demo.py galaxy_collision_3d --profile local --frames 40 --backend cpu
```

Completed Star in a Bottle runs expose a **Rotate / field** control in the web
viewer. Drag to rotate the computed torus, use the wheel to zoom, and switch
between the final plasma-flow state and an explicitly labelled illustrative
magnetic-confinement geometry view.

Generated runs are stored below `runs/`. The viewer can replay any run without recomputing it.

### Deep zoom

After a crystal run completes, the viewer's **Deep zoom** button switches to a
canvas that re-renders the crystal's geometry for whatever window is on screen.
Magnification is effectively unbounded — thousands of times in is normal, and
new branch generations keep appearing because the growth rule is applied at
every scale. Scroll to zoom at the cursor, drag to pan.

Aim at a branch, not at empty space: a fractal is mostly void, and zooming into
a gap shows nothing. Aiming at the *middle* of a thick trunk eventually fills
the view with solid white, so steer toward edges and tips.

Runs also bake a shallow tile pyramid into `runs/<id>/zoom/`, which paints
instantly and keeps a finite zoom working on a machine with no solver — for a
run synced from Leonardo. See `docs/ARCHITECTURE.md` to add it elsewhere.

### PyTorch

The neural-network wall trains real networks when PyTorch is present and falls
back to a clearly-labelled non-training surrogate when it is not:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For convenience there are also individual scripts in `scripts/run_*.py`.

## Compute: CPU or GPU

The viewer has a **Compute** selector beside the profile:

- `Auto` - use CUDA if a working CuPy/PyTorch device is present, else CPU.
- `CPU` - force NumPy (and force PyTorch onto the CPU for the neural wall).
- `GPU` - require CUDA; shown as unavailable, with the reason, when there is none.

The header pill reports what the machine can actually do. From the command line
use `--backend`, or set the environment variable, which is what the SLURM job
does:

```bash
python run_demo.py fluid --profile desktop --backend gpu
LEONARDO_DEMO_BACKEND=cupy python run_demo.py fluid --profile leonardo
```

`--backend hybrid` keeps the solver on CUDA while two CPU workers encode and
write frames in parallel. It is useful on a machine with one GPU as well as on
Leonardo; it is a real pipeline, not two devices concurrently updating one
array. See [`docs/HYBRID_COMPUTE.md`](docs/HYBRID_COMPUTE.md).

The galaxy collision also has a true shared-memory CPU path: NumPy updates
disjoint tracer chunks across the cores assigned through
`SLURM_CPUS_PER_TASK`. Its CUDA path keeps the same leapfrog state on the GPU
between frames. Backend and allocated CPU worker count are recorded in
`meta.json`.

The separate `galaxy_collision_3d` demo follows the direct tiled all-pairs
force strategy from NBody-EuroHPC. Every disc, bulge, and dark-halo
super-particle carries mass and evolves in three dimensions. Its Milky Way
disc is conditioned by a reduced Gaia DR3 3-D sample; its M31 disc uses PHAT
v3 sky positions deprojected with the documented M31 geometry. PHAT does not
measure useful individual stellar depths at M31, so the vertical thickness is
explicitly modelled and recorded rather than presented as observed data.

Whatever is chosen is recorded in each run's `meta.json` and shown in the
viewer, so a replayed run always says what computed it.

## Saved runs and replay

Every run is written to `runs/<id>/` and stays there. The **Saved runs** library
on the front page lists them with thumbnails; each demo page also shows replay
chips for its own history. Opening one restores its parameters, profile and
backend and enables playback, reveal and deep zoom straight from disk, with no
recomputation.

This is the same path `docs/LEONARDO.md` recommends as the event-day fallback:
generate good runs in advance, and replay one if a live job queues. Label it as
playback when you do.

## Profiles

- `local`: deliberately small; intended for rapid visual iteration on a CPU.
- `desktop`: larger defaults; uses CUDA automatically if CuPy/PyTorch is installed and working.
- `leonardo`: larger defaults, headless output, and settings intended for SLURM jobs. Start smaller and benchmark before increasing resolution.

All profile numbers are in `config/profiles.json` and can be edited without modifying the simulation code.

## Leonardo workflow

The recommended exhibition workflow is:

```text
visitor / presenter PC
        |
        | parameter JSON / job request
        v
Leonardo login node -> SLURM -> DCGP CPU or Booster GPU node
                                  |
                                  | PNG frames + metadata
                                  v
                                $FAST
                                  |
                             rsync/scp/sftp
                                  |
                                  v
                         exhibition PC viewer
```

The repository contains:

- `slurm/run_demo.sbatch` - one-GPU Booster job template.
- `slurm/run_demo_cpu.sbatch` - shared-memory DCGP CPU job template.
- `config/leonardo.env.example` - account, environment, QoS and output settings.
- `scripts/submit_leonardo.sh` - CPU/GPU/hybrid-aware submission wrapper.
- `scripts/sync_run_from_leonardo.sh` - incremental result synchronisation.
- `tools/leonardo_preflight.py` - allocation, dependency, CUDA and output check.
- `docs/LEONARDO.md` - detailed setup and event-day fallback plan.
- `docs/NBODY_REFERENCE_AUDIT.md` - comparison with the referenced MUrB code and explanation of its blue pulse.
- `docs/LEONARDO_DEMO_AUDIT.md` - per-demo backend audit, measured stage timings, and benchmark reproduction.
- `benchmarks/index.html` - interactive CPU/GPU/hybrid timing report (also served at `/benchmarks/`).

The event should always have a **playback fallback**. If a live job queues, a connection drops, or the venue network is unreliable, the viewer can switch to a previously generated run while clearly labelling it as playback.

## Scientific scope

These are **public-engagement demonstrators**, not drop-in replacements for production research codes. The repository explicitly labels approximations in each demo's scientific note. The PBH module in particular includes an adapter for a validated external solver rather than pretending the reduced exhibition model is full numerical relativity.

Read `docs/SCIENTIFIC_NOTES.md` before presenting results as quantitative science.

## Recommended build order

If you are modifying the project, work in this order:

1. `reaction_diffusion` - quickest test of the full compute -> frames -> viewer pipeline.
2. `neural_wall` - validates the ensemble reveal.
3. `black_hole` - flagship pixel-parallel GPU story.
4. `pbh` - connect the real research solver.
5. `crystal` - reuses PDE visualisation infrastructure.
6. `galaxy_collision`.
7. `cosmic_web`.
8. `fluid` - scale up only after benchmarking.
9. `fusion_plasma` - validates torus projection and nonlinear-wave ensembles.
10. `weather_ensemble` - validates forecast divergence on the globe.
11. `molecular_dynamics` - validates all-pairs scaling and 3-D particle rendering.

See `docs/ARCHITECTURE.md` and `docs/DEMO_STORIES.md` for the full design.

## Adding a demo

Read **`docs/DEMO_TEMPLATE.md`** first — it is the house style, and every rule
in it exists because breaking it silently produced a demo that looked finished
and was not. Copy `leonardo_demos/demos/_template.py` as a starting point.

**`docs/IMPROVEMENTS.md`** records what changed from the first version of this
repository and why, including the bugs that motivated each rule.
