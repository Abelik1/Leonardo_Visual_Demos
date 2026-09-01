# Black-hole lensing

## Purpose

Shows the same lensing calculation from two useful viewpoints: the clean 2-D
observer image and a fixed-camera 3-D view of numerically advanced photon paths.

## Implementation map

- Solver and renderers: `leonardo_demos/demos/black_hole.py`
- Parameters: `mass`, `spin`, `disk`
- Main output: `frames/` (2-D observer image)
- Alternative output: `modes/3d/` (ray-space animation)
- Viewer mode controls: `web/app.js`

## Scientific boundary

The image mapping and 3-D paths are weak-field exhibition models. The path
integrator renormalises photon direction and includes a qualitative signed
spin term; it is not a Kerr null-geodesic solver or GRMHD calculation.

## Extension points

Replace `lens()` and `integrate_rays()` together for a validated relativistic
backend. Preserve both output modes and publish method details through metadata
and HTML layers rather than drawing them into frames.
