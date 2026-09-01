# Galaxy collision

## Purpose

Shows a restricted N-body Milky Way–Andromeda encounter using physical masses,
distances, velocities, and a camera that follows the evolving system.

## Implementation map

- Solver and renderer: `leonardo_demos/demos/galaxy_collision.py`
- Optional reduced M31 catalogue: `data/m31_catalog_reduced.npz`
- Parameters: preset, impact, speed, tilt
- Reveal: a transverse-velocity uncertainty sweep

## Scientific boundary

Galaxy centres drive the restricted potential while tracer stars expose tidal
structure. This is not a self-consistent live dark-matter simulation; use the
separate `galaxy_collision_3d` demo for direct super-particle gravity.

## Units

Working units are kpc, km/s, and solar masses. Simulation time converts using
1 kpc/(km/s) = 0.97779 Gyr; retain that conversion in displayed values.
