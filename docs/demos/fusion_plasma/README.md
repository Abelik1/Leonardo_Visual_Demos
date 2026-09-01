# Star in a Bottle

## Purpose

Evolves a reduced nonlinear plasma-wave field on a periodic lattice, maps that
state onto a torus, and advects passive tracers through its derived drift.

## Implementation map

- Solver and frame renderer: `leonardo_demos/demos/fusion_plasma.py`
- Interactive 3-D renderer: `web/fusion_view.js`
- Parameters: magnetic field, heating, density
- Main frames: clean pre-rendered torus views
- Live 3-D state: `modes/fusion3d/frame_NNNN.json`
- Legacy final-state fallback: `fusion_view.json`

## Viewer behaviour

The rotatable 3-D canvas is the default and consumes one state per playback
frame. Plasma-flow and magnetic-geometry layers can be changed without stopping
the timeline or resetting the visitor's camera.

## Scientific boundary

Magnetic lines are illustrative helical confinement geometry, not a solved
tokamak equilibrium, q-profile, MHD equilibrium, or disruption calculation.
