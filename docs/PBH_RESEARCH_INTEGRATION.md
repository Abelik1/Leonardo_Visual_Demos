# Integrating the full PBH research solver

The final event version should use the validated PBH solver from the research project rather than the reduced exhibition model wherever possible.

## Minimal interface needed by the viewer

For each saved time `t_j`, export:

- `r`: radial coordinate array;
- `rho_over_rhob`: density relative to FRW background;
- `compaction`: `C(r)`;
- optionally `gamma` and `u_over_gamma`;
- scalar metadata: `delta`, threshold/reference value, simulation time, solver resolution.

One convenient format is:

```text
pbh_run/
  metadata.json
  state_0000.npz
  state_0001.npz
  ...
```

Each NPZ can contain the named NumPy arrays above.

Then run:

```bash
python tools/render_pbh_research.py /path/to/pbh_run --out runs/pbh_research
```

The resulting folder follows the same `frames/ + meta.json + reveal.jpg` contract as all other demos.

## Ensemble computation

The most natural HPC extension is not to make one spherical run absurdly large. Batch independent parameter values:

```text
(delta, profile width, q, equation-of-state parameters, numerical resolution, ...)
```

Each rank/GPU can execute a different batch. The exhibition reveal then uses the **actual final states** to construct a phase diagram or a wall of collapse/dispersion outcomes.
