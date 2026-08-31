# Troubleshooting

## `ModuleNotFoundError`

Activate your venv and install `requirements.txt`.

## CuPy fails to import

Run with NumPy first:

```bash
export LEONARDO_DEMO_BACKEND=numpy
```

Then install a CuPy build matching the CUDA runtime of the actual machine.

## Neural wall says `numpy-fallback`

PyTorch is not installed/usable. The rest of the viewer still works. Install a CUDA-compatible PyTorch build before presenting it as real neural-network training.

## Viewer shows a broken image while a run is live

The poller may have requested a frame during the tiny interval while it was still being saved. The next poll normally fixes it. For maximum robustness, save to a temporary filename and rename atomically; this is a good production hardening task.

## A Leonardo job sits in the queue

Use `scontrol show job JOBID` to inspect the pending reason. Do not make the exhibition dependent on immediate scheduling; keep cached playback runs.

## The local fluid demo is slow

Lower `fluid.nx`, `fluid.ny`, or `steps_per_frame` in `config/profiles.json`. The purpose of the local profile is visual development, not final resolution.


## Every demo fails immediately, with a KeyError in the traceback

Almost always a **stale viewer**. The server imports the demo modules once at
startup but re-reads `config/profiles.json` on every request, so a viewer left
running from before an update executes old code against new settings and every
run dies on a missing key. Tracebacks from a stale process are a giveaway: the
line numbers point at comments or docstrings, because Python captured them
against the old file and prints source from the new one.

Close every viewer window and start one fresh. `app.py` now moves to the next
free port if 8000 is taken and prints a warning, rather than failing to bind
and leaving the browser pointed at the old server.

## The viewer starts but PyTorch is missing

Launch it with `scripts/start_viewer_windows.bat`, which uses `.venv`. Running
`python app.py` against whatever interpreter is on PATH picks up a different
environment without the installed packages.
