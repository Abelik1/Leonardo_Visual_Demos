# Cosmic-web formation

## Purpose

Evolves Zel'dovich initial conditions with particle-mesh gravity so clusters,
filaments, and voids emerge from small primordial fluctuations.

## Implementation map

- Solver and renderer: `leonardo_demos/demos/cosmic_web.py`
- Parameters: seed, gravity, helium fraction, expanding space, dark energy,
  and warm-dark-matter cutoff
- Reveal: selectable number of independent gas-composition runs

## Selectable rules

- Expanding space enables comoving evolution and Hubble damping.
- Dark energy adds a qualitative late-time exponential acceleration term and
  only applies while expansion is enabled.
- Warm dark matter applies a qualitative short-wavelength transfer cutoff to
  the initial power spectrum.
- Helium changes mean molecular weight and therefore the Jeans support scale.

These switches compare formation ideas visually; they are not precision
Boltzmann-code predictions or cosmological parameter constraints.
