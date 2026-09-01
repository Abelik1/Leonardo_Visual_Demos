# Primordial black-hole threshold

## Purpose

Makes critical collapse legible: a small change in the initial density contrast
switches a radial perturbation between dispersion and rapid concentration.

## Implementation map

- Solver and renderer: `leonardo_demos/demos/pbh.py`
- Parameters: `delta`, `width`
- Profile controls: `radial_points`, `ensemble`
- Reveal: independent samples around the reference threshold and profile width

## Scientific boundary

This is a reduced qualitative radial model. It visualises threshold behaviour
but does not reproduce the full Misner-Sharp/Chebyshev numerical-relativity
solver described by the research capstone.

## Extension points

Validated research output should enter through the existing run/frame contract.
Preserve the explicit distinction between the exhibition model and research
results, and keep threshold/readout values in frontend overlays.
