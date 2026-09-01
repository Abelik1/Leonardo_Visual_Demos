# Agent notes: virtual wind tunnel

- The visible bodies must match the solver's bounce-back mask exactly.
- Apply preset and custom blocks before initialising the LBM state.
- Use the same mask for tracer collision; prevent particles crossing solids.
- Keep the inlet and outlet open and validate custom grid dimensions server-side.
- Do not tie total lattice updates directly to the number of saved frames.
- Verify all three presets plus at least one custom painted block arrangement.
