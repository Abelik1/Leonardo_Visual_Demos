# Agent notes: black-hole lensing

- Read `../../../leonardo_demos/demos/AGENTS.md` and this guide first.
- Preserve `lens(bg, mass, spin, t=...)` compatibility used by tests and tools.
- Keep 3-D paths numerical; do not replace them with decorative curves.
- Maintain a visible mix of escaping and captured rays across normal controls.
- Keep 2-D in `frames/`, 3-D in `modes/3d/`, and default the viewer to 3-D.
- Describe the model as weak-field unless a validated GR integrator replaces it.
- Verify one frame from each mode and run `tests/test_small_demos.py`.
