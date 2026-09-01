# Crystal growth

## Purpose

Builds recursive anisotropic branch geometry for several crystal habits and
supports effectively unbounded visual zoom by regenerating detail on demand.

## Implementation map

- Demo orchestration: `leonardo_demos/demos/crystal.py`
- Geometry and rasterisation: `leonardo_demos/crystal_growth.py`
- Deep-zoom API and cache: `app.py`
- Browser viewer: `web/deepzoom.js`
- Parameters: undercooling, anisotropy, symmetry, habit mode, seed

## Deep-zoom contract

Zoom requests describe a mathematical viewport. The server regenerates branch
geometry at the required level and returns coherent cached tiles/viewports; it
does not magnify the original JPEG.

## Performance

Recursive geometry is CPU-heavy and independently generated habits use process
parallelism. Keep segment budgets explicit and deterministic.
