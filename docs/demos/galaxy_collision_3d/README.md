# Galaxy collision — full 3-D gravity

## Purpose

Runs a softened direct-force encounter in which every visible disc, bulge, and
dark-halo super-particle contributes to the gravitational acceleration.

## Implementation map

- Solver and JPEG renderer: `leonardo_demos/demos/galaxy_collision_3d.py`
- Interactive renderer: `web/galaxy3d_view.js`
- Parameters: impact, speed, disc tilt, softening
- Main frames: `frames/`
- Rotatable state: one JSON file per frame under `interactive/`

## Data and scope

Gaia and PHAT-derived samples condition the visible starting morphology, while
the reduced particle system remains an illustrative super-particle experiment,
not a fitted equilibrium prediction of the Local Group.

## Performance

Force work scales quadratically with particle count. CPU calculation is tiled
and may use assigned workers; CUDA uses the device array implementation.
