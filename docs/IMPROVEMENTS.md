# What changed, and what it taught us

A record of the substantive changes made to the first version of this
repository. It is written to be read alongside `docs/DEMO_TEMPLATE.md`: most of
the rules there are generalisations of a specific bug listed here.

## Viewer

**Compute selection.** The run API always accepted a `backend`, but nothing
exposed it. There is now a **Compute** selector (Auto / CPU / GPU) beside the
profile. `backend.probe()` reports what the machine can actually do, so the GPU
option is disabled with the reason attached rather than silently falling back —
an option that lies is worse than one shown as unavailable. The header pill
states the machine's capability.

`neural_wall` previously took CUDA whenever PyTorch could see it, ignoring the
request entirely. Device selection now goes through
`backend.torch_device(ctx.backend_requested)`, so a CPU run really is a CPU run
and the reported backend is true.

On the Windows exhibition workstation the selector was initially unavailable
despite an RTX 3060 Ti being present: Python had a CPU-only PyTorch build and
no CuPy package. The verified project-local setup is now CUDA 13 (`torch
2.13.0+cu130` and `cupy-cuda13x 14.2.0`); the NVIDIA driver supplies the device
and no system-wide CUDA toolkit is required. See `requirements-gpu.txt`.

**Saved runs.** Every run has always persisted to `runs/<id>/`, but there was
no way to reach one: `/api/runs` existed and nothing called it. There is now a
**Saved runs** library with thumbnails, and per-demo replay chips on the stage.
Replaying restores the parameters, profile and backend, and enables playback,
reveal and deep zoom — all from disk, with no recomputation.

This matters beyond convenience: `docs/LEONARDO.md` makes playback the event-day
fallback, and until now that fallback had no button.

**Deep zoom.** The old zoom applied a CSS transform to the finished JPEG, so it
could only ever magnify pixels. Now:

- `/api/zoom/{run}` regenerates the model's geometry for exactly the window on
  screen. No level count, so no depth limit — thousands of times in is normal.
- A pre-baked tile pyramid still ships with each run and paints instantly, and
  is the only source when no solver is present (a run synced from Leonardo).

Getting genuinely unbounded zoom needed the model to have detail at *every*
scale. A segment between two branch points was a bare straight line, so
magnifying there showed a line forever however deep the recursion ran. Every
segment now also spawns half-length *carriers* that re-apply the branching rule
at half scale, recursively.

**Panning.** The deep-zoom canvas is a child of `.screenWrap`, which still
carried the legacy zoom handlers. Pointer events bubbled into them, the parent
took pointer capture, and `pointerup` never reached the canvas — leaving the
view panning with no button held and no way to stop. The legacy handlers are
disabled while deep zoom is active, and the drag now ends on
pointerup/cancel/leave/blur at window level too.

## Robustness

**Stale viewers.** The server imports demo modules once at startup but re-reads
`config/profiles.json` on every request, so a viewer left running across an
update executes old code against new settings and every run dies on a missing
key. `app.py` now moves to the next free port when 8000 is taken and prints a
warning, instead of failing to bind and leaving the browser talking to the old
process. `scripts/start_viewer_windows.bat` uses `.venv` rather than whatever
`python` is on PATH.

Diagnostic worth remembering: a traceback whose line numbers point at comments
or docstrings is a stale process. Python captured the numbers against the old
file and printed source from the new one.

## Demos

**reaction_diffusion.** Ran ~1000 solver steps where Gray-Scott needs O(10⁴);
the screen showed nine dots on black because the pattern had barely left its
seeds. Now runs a `total_steps` budget on a 16:9 domain, so the spots are round
and the parameter sweep spans labyrinths, spots and an extinction boundary.

**neural_wall.** PyTorch was absent, so it silently ran a Fourier surrogate
under the headline "This network is learning…". The surrogate also clipped a
shared progress value at 1.0, so all 128 networks became byte-identical within a
fifth of the run and the reveal was a wall of duplicates with the same loss
printed on every tile. Now: real batched training (stacked weight tensors, one
forward/backward, per-network learning rates via a hand-written Adam, widths
varied by masking), laid out as a 2-D hyperparameter grid, with the surrogate
labelled as *not training* wherever it appears.

It is now an RGB coordinate painter: each network maps `(x, y)` to `(red,
green, blue)`. Visitors can choose an RGB preset, draw a shape, upload a photo,
or explicitly allow a live camera preview and capture a frame. The browser
reduces that image to the small learning target and sends it only when a run is
started. A scan-head overlay makes the progressive reconstruction legible, and
the side panel exposes the real winning weights and loss curve. The Windows
CUDA path also imports PyTorch before CuPy for this demo; this avoids a
first-call deadlock with their bundled CUDA runtimes.

**black_hole.** The previous opening jumped straight from an unlensed sky to a
flat image. It now starts with a fixed-camera exploded optical stack: foreground
dust, distant source sheet, gravitational lens and observer image plane are
separate; visible light packets curve around the lens; then the layers compact
and dissolve into the real pixel-parallel lensing output. This is intentionally
labelled as an explanatory layer view — the animated paths are not claimed to
be individually integrated Kerr geodesics.

**crystal.** Was a phase-field grid whose "zoom" could only enlarge pixels. Now
a recursive geometric model with six habits (classic, fern, seaweed, star,
coral, plate) and selectable 3–12-fold symmetry, generated lazily and pruned to
the view. Frames and zoom come from one generator, so magnifying a frame cannot
reveal a different crystal.

**fluid.** Displayed |vorticity| on a fire palette, which read as a heat map.
Now shows flow speed on a cool palette with advected streaklines carrying a
position history, direction glyphs, and live stagnation/wake pressure. A
half-cell offset and a little transverse noise break the symmetry so the wake
actually develops.

**galaxy_collision.** Now runs the real Milky Way / Andromeda encounter in kpc,
km/s and solar masses, with an honest clock in Gyr. Three bugs on the way:
circular velocities were computed from a point-mass potential while the force
used a softened one (inner stars at ~2000 km/s, discs exploding); the relative
velocity sign was inverted, so the galaxies moved apart; and the camera framed
on a per-coordinate percentile, which is dominated by small y values and clipped
both galaxies off the edges.

The starting discs now make the intended morphology readable at exhibition
scale: four Milky Way arms, and M31's two arms plus ring, with live close-up
windows made from the same simulated tracers. Rendering first bins each
tracer's represented galaxy mass, then uses the two density fields for the
blue Milky Way, warm M31 and bright overlap colours. A curator can also install
`data/m31_catalog_reduced.npz`; `tools/reduce_star_catalog.py` turns an
offline, deprojected catalogue into mass-weighted spatial representatives so a
catalogue-driven M31 start remains bounded and repeatable.

**cosmic_web.** Never produced a cosmic web at all — it collapsed the box into a
single blob. Three separate causes: the Poisson source was raw particle counts
rather than the dimensionless contrast δ; `fftfreq(n)*2π` gave k in radians per
*cell* while positions are in box units, making forces n=128× too strong; and
random position jitter put nearly all the power in the box-scale mode. It now
uses Zel'dovich initial conditions from a power spectrum and integrates on an
expanding background, and produces filaments, nodes and voids.

It also carries gas composition: hydrogen A=1.008 and helium-4 A=4.003 — note
that helium's atomic *number* is 2 but its *mass* number is 4, and mass is what
sets the dynamics. µ = 1/(X/1.008 + Y/4.003) runs from 1.008 (pure H) through
1.229 (primordial) to 4.003 (pure He), and a heavier µ means a shorter Jeans
length and finer structure.

## Recurring lessons

1. **Work budgets are physics, not presentation.** Two demos looked broken
   purely because their step counts were derived from the frame count.
2. **Aspect ratio is a correctness issue.** Stretching a square field to 16:9
   silently falsified the geometry in three demos.
3. **Check units in spectral solvers.** `fftfreq` returns cycles per *sample*.
   Two separate factor-of-n errors came from that.
4. **Normalisation hides in helpers.** Autoscaling in a palette turned a
   correct solid-white render into solid background.
5. **Truncation must be symmetric.** Depth-first generation under a budget
   produces lopsided results that read as bugs in the model.
6. **Verify by looking.** Every one of these was found by rendering the output
   and examining it, not by reading the code or trusting a green test.
