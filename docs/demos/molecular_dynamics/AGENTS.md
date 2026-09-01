# Agent notes: Molecular Machine

- Describe the model as coarse-grained, not atomistic.
- Keep bonded and non-bonded force signs and exclusions physically consistent.
- Maintain stable integration across the complete temperature/control range.
- Treat particle-count increases as O(N²) performance changes.
- Honour `_parallel_count` for independent reveal trajectories.
- Verify finite coordinates, bond continuity, and distinct reveal conditions.
