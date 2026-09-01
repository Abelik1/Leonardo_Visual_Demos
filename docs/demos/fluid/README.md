# Virtual wind tunnel

## Purpose

Shows velocity, streaklines, pressure response, and vortex shedding around solid
geometry using a D2Q9 lattice-Boltzmann flow solver.

## Implementation map

- Solver and renderer: `leonardo_demos/demos/fluid.py`
- Parameters: `speed`, `obstacle`
- Presets: single cylinder, twin cylinders, ellipse plus block
- Custom input: validated 24×12 browser obstacle grid in `_obstacle_grid`
- Frontend builder: `web/app.js` (`addFluidBuilder`)

## Obstacle contract

`build_obstacles()` creates the exact boolean mask used for bounce-back,
tracer collision, pressure sampling, and drawing. Custom cells are combined
with the selected preset; the browser prevents painting inlet/outlet edge cells.

## Performance

Vortex shedding needs a substantial fixed lattice-step budget. Frame count is
presentation resolution and must not silently shorten or multiply the physics.
