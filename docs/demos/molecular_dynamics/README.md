# Molecular Machine

## Purpose

Evolves a coarse-grained 3-D molecular chain under bonded, repulsive, attractive,
thermal, and solvent-dependent forces.

## Implementation map

- Solver and renderer: `leonardo_demos/demos/molecular_dynamics.py`
- Parameters: temperature, attraction, solvent quality, sequence
- Main frames: depth-sorted particles and bonds
- Reveal: independent trajectories across laboratory conditions

## Scientific boundary

Each bead is coarse-grained and can represent more than one atom or residue.
The demo illustrates competing molecular forces and conformational change; it is
not an all-atom force field, protein-folding predictor, or free-energy result.

## Performance

Non-bonded work is all-pairs and scales quadratically with particle count. Keep
integration stable before raising particle counts or timestep sizes.
