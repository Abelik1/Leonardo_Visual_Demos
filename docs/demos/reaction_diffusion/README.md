# Living mathematics

## Purpose

Evolves the Gray-Scott reaction-diffusion equations from a small disturbance to
produce spots, waves, labyrinths, and other emergent patterns.

## Implementation map

- Solver and renderer: `leonardo_demos/demos/reaction_diffusion.py`
- Parameters: feed and kill rates
- Profile controls: grid size, total steps, sweep steps, ensemble
- Reveal: neighbouring feed/kill parameter simulations

## Numerical behaviour

The two chemical fields use local reaction terms and finite-difference
diffusion. Total solver work is fixed by profile so changing the saved frame
count changes temporal sampling, not the final maturity of the pattern.

## Extension points

A drawing/seed editor can be added as a validated initial-condition payload,
following the neural target and fluid-grid request patterns.
