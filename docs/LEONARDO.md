# Running on Leonardo

This repository was designed around the workflow described in the supplied CINECA Leonardo introduction: log in through CINECA access, submit long work to compute nodes with **SLURM**, use the GPU-accelerated **Booster** for CUDA workloads, and keep large transient output in the appropriate shared filesystem rather than trying to render on the login node.

## 1. Account and access

Before event day confirm:

- your HPC username works;
- the event project account is active;
- the account has Booster budget;
- the exact project-account string for `#SBATCH --account=...`;
- the allowed `boost_usr_prod` / QoS limits for the event;
- which datamover/login hostname the organisers want you to use for transfers.

The example scripts default to `EUHPC_TDEMO_26` because it appears in the supplied training slides. Change it through:

```bash
export LEONARDO_ACCOUNT=YOUR_REAL_ACCOUNT
```

Do not assume that the training account name or its expiry applies to the actual event.

## 2. Prepare the Python environment

### Safe first test

Use NumPy only. This verifies paths, SLURM, output and synchronisation before CUDA dependencies are involved.

```bash
python -m venv $WORK/venvs/leonardo-demos
source $WORK/venvs/leonardo-demos/bin/activate
pip install -r requirements.txt
```

Internet access/pip availability can differ on HPC systems. If package download is unavailable, build the venv/container ahead of time using the CINECA-recommended software stack or ask support which Python/FastAPI/Pillow modules are available.

### GPU mode

For CuPy/PyTorch, install builds compatible with the CUDA stack active on Leonardo. Do **not** copy a random local CUDA wheel and assume ABI compatibility.

Test:

```bash
python - <<'PY'
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())
a=cp.arange(10); print((a*a).get())
PY
```

Then:

```bash
export LEONARDO_DEMO_BACKEND=cupy
```

### GPU + CPU pipeline

Use `LEONARDO_DEMO_BACKEND=hybrid` when one GPU should evolve the model while
the host CPU encodes the presentation frames and writes them to disk in
parallel. The provided Booster job already requests one GPU and eight CPU
cores, which is the correct single-node shape for this mode. See
[`HYBRID_COMPUTE.md`](HYBRID_COMPUTE.md) for the exact division of work and
the distinction between a Booster CPU and a separate DCGP CPU-only job.

## 3. Submit one demo

```bash
export LEONARDO_ACCOUNT=YOUR_ACCOUNT
export LEONARDO_RUNROOT=$FAST/leonardo_visual_demos
bash scripts/submit_leonardo.sh reaction_diffusion 90 leonardo
```

Or directly:

```bash
sbatch --account=$LEONARDO_ACCOUNT --export=ALL,DEMO=black_hole,FRAMES=90,PROFILE=leonardo slurm/run_demo.sbatch
```

Monitor:

```bash
squeue --me
scontrol show job JOBID
```

Cancel:

```bash
scancel JOBID
```

## 4. Transfer frames while the job is running

On the presentation PC, synchronise the run directory every few seconds. `rsync` naturally skips frames already copied.

```bash
rsync -av USER@YOUR_CINECA_TRANSFER_HOST:/remote/run/path/ ./runs/leonardo_live/
```

You can then refresh the local viewer or place the synced folder directly under `runs/`.

A future controller can automate this polling. For the first event version, keeping transfer and rendering loosely coupled is more robust.

## 5. Playback fallback

Generate at least one high-quality Leonardo run of every demonstration the day before the event. Keep two copies:

- on the event PC;
- on a USB/portable SSD.

If a queue is long or connectivity fails, switch to playback and say clearly:

> "This is a run computed earlier on Leonardo; we're replaying the saved simulation states."

Do not imply that a cached run is live.

## 6. Do not compute on login nodes

The login node should be used for file preparation, compilation/environment setup, small checks and submission. The long simulations belong in SLURM jobs on compute nodes.

## 7. Scale gradually

The `leonardo` profile is a **starting target**, not a promise that every workload is optimally sized. Benchmark one demo at a time:

1. one GPU, small resolution;
2. one GPU, final single-GPU resolution;
3. several independent ensemble tasks;
4. only then introduce MPI/domain decomposition where it materially improves the demo.

The strongest event story is often *many simulations at once*, not one enormous simulation that spends most of its time communicating.
