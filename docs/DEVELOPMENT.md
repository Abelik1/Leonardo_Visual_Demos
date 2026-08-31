# Development guide

## Adding a demo

`docs/DEMO_TEMPLATE.md` is the house style and the checklist;
`leonardo_demos/demos/_template.py` is a skeleton to copy.
`docs/IMPROVEMENTS.md` explains why each rule is there.

## Add a new visual effect without breaking the science

Keep these layers separate:

1. **simulation state** - arrays/particles/fields;
2. **headless renderer** - converts state to a frame;
3. **story overlay** - titles, counters, reveal text;
4. **web transitions** - fades/zoom/replay controls.

Never change a scientific state merely because a camera transition looks better. Change how it is rendered.

## Performance workflow

Use `local` until the story is visually convincing. Then measure:

```bash
python run_demo.py DEMO --profile desktop --frames 20
```

Only then test `leonardo`.

For array demos, forcing CPU/GPU is useful. The CLI flag, the environment
variable and the viewer's Compute selector all feed the same request:

```bash
python run_demo.py reaction_diffusion --profile desktop --frames 20 --backend cpu
python run_demo.py reaction_diffusion --profile desktop --frames 20 --backend gpu
LEONARDO_DEMO_BACKEND=cupy python run_demo.py reaction_diffusion --profile desktop --frames 20
```

Check what the machine actually offers:

```bash
python -c "from leonardo_demos.backend import probe; import json; print(json.dumps(probe(), indent=2))"
```

A demo that uses another framework must route its device through
`backend.torch_device()` so a CPU run really stays on the CPU.

## Frame count

For event playback, 60–120 scientific states is normally enough. The browser can transition between them smoothly. Increasing solver resolution is usually more meaningful than writing 1,000 nearly identical JPEGs.

## Testing

Run:

```bash
python -m unittest discover -s tests -v
```

The smoke tests intentionally use tiny profiles and only verify that output contracts remain valid.
