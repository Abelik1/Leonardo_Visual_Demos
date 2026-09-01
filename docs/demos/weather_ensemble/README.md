# Storm Factory

## Purpose

Evolves a reduced rotating atmosphere with barotropic vorticity and advected
moisture, then shows how small initial uncertainties create different futures.

## Implementation map

- Solver and globe renderer: `leonardo_demos/demos/weather_ensemble.py`
- Parameters: warming, jet-stream strength, uncertainty
- Main frames: one evolving forecast globe
- Reveal: independent perturbed initial-condition forecasts

## Scientific boundary

This is an exhibition-scale atmosphere, not an operational numerical weather
prediction model. The controls alter reduced dynamics and ensemble spread, not
a calibrated real-world forecast.

## Parallel reveal

The selected parallel count controls how many admissible perturbed forecasts
are computed. Increasing it samples uncertainty more densely and raises compute
and rendering cost.
