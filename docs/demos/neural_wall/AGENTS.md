# Agent notes: neural-network wall

- The network input is coordinates and output is RGB; do not describe it as recognition.
- Keep target, weights, loss chart, and policy graph out of the main image stream.
- Store custom targets inside the run and validate PNG size before decoding.
- Preserve true PyTorch training labels; never disguise the surrogate as training.
- `_parallel_count` controls the number of networks and must remain GPU-batch friendly.
- Verify a preset, browser drawing, uploaded/captured RGB path, and network overlay.
