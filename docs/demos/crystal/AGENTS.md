# Agent notes: crystal growth

- Deep zoom must regenerate geometry rather than scale a bitmap.
- Keep geometry deterministic for the same seed and viewport.
- Bound recursion with segment budgets and profile depth.
- Preserve process-based CPU parallelism; Python threads do not solve the GIL cost.
- Honour the selected reveal count and handle non-default grid sizes safely.
- Verify a normal frame, reveal, and at least two deep-zoom levels.
