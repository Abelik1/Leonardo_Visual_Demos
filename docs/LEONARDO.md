# Running on Leonardo

The compute side of this project is headless: it writes JPEG frames and JSON
metadata without OpenGL, X11, or a web server. This is the right shape for
Leonardo's batch nodes and remains compatible with CINECA's current mitigation
that disables Xorg on Booster GPU nodes.

The two supported allocations are intentionally separate:

| Mode | Partition | Job shape | Implementation |
|---|---|---|---|
| CPU | `dcgp_usr_prod` | 1 process, 32 CPU cores | NumPy plus chunked shared-memory tracer updates |
| GPU / hybrid | `boost_usr_prod` | 1 process, 1 A100, 8 CPU cores | CuPy solver; hybrid overlaps CPU JPEG encoding |

Leonardo Booster nodes have 32 CPU cores and four 64 GiB A100 GPUs. DCGP nodes
have 112 CPU cores and no GPUs. This demo asks only for the slice it can use;
it does not reserve a whole DCGP node or four A100s merely to make the resource
request look larger.

## 1. Configure the account and paths

On Leonardo, copy the example and edit the real project-account value:

```bash
cp config/leonardo.env.example config/leonardo.env
saldo -b
```

`config/leonardo.env` is ignored by Git. The submission wrapper requires an
explicit `LEONARDO_ACCOUNT`; it will not silently charge a training account
copied from a slide deck. Confirm that the account is entitled to the selected
partition and QoS.

Output defaults to `$FAST/leonardo_visual_demos`. `$FAST` is shared and suited
to the frame stream. Booster nodes are diskless and expose only a fixed 10 GiB
RAM-backed temporary area, so the job does not stage a complete run in
`$TMPDIR`.

## 2. Prepare one tested Python environment

CPU-only setup:

```bash
module purge
module load profile/base
python3 -m venv "$WORK/venvs/leonardo-visual-demos"
source "$WORK/venvs/leonardo-visual-demos/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the CuPy demos on Booster, load Leonardo's CUDA 12.2 module and install the
CUDA 12 wheel family:

```bash
module load cuda/12.2
python -m pip install -r requirements-gpu.txt
```

Do not install the CUDA 13 Windows lock on Leonardo. `requirements-gpu.txt`
contains CuPy for the CUDA 12 family; `requirements-gpu-windows.txt` is only
for the separately tested exhibition workstation. The neural-network wall is
the sole PyTorch demo. If it must train on a Leonardo GPU, prefer a CINECA
`cineca-ai` module and create the environment with `--system-site-packages`.

An alternative is a pre-built Singularity image. See `container/README.md` and
set `LEONARDO_CONTAINER` instead of `LEONARDO_VENV`.

## 3. Submit and validate

The fourth wrapper argument chooses the allocation and backend:

```bash
# DCGP shared-memory CPU run
bash scripts/submit_leonardo.sh galaxy_collision 90 leonardo cpu

# Booster: solver on one A100
bash scripts/submit_leonardo.sh galaxy_collision 90 leonardo gpu

# Booster: A100 solver plus overlapping CPU frame encoding
bash scripts/submit_leonardo.sh galaxy_collision 90 leonardo hybrid
```

The wrapper selects `slurm/run_demo_cpu.sbatch` for CPU work and
`slurm/run_demo.sbatch` for GPU/hybrid work. Both bind the process to its
allocated cores, propagate `SLURM_CPUS_PER_TASK`, write a unique job-id run
directory, and execute `tools/leonardo_preflight.py` before the simulation.
The preflight verifies imports, output writability, and—on Booster—the actual
CuPy-visible device. Explicit GPU requests fail rather than silently falling
back to NumPy.

The example config uses the 30-minute debug QoS values. The job templates also
request 30 minutes. For production runs, select a QoS allowed for the project
and change the time limit in the job file if necessary.

Monitor with:

```bash
squeue --me
scontrol show job JOBID
```

## 4. CPU and GPU scope

The collision is a restricted N-body model: each tracer feels the two moving
galaxy potentials, so its cost is O(N), not the O(N²) all-pairs benchmark in
the reference MUrB repository. On NumPy, disjoint tracer chunks execute in
parallel using the cores granted to the process. On CuPy, all tracer state
stays on the A100 through the leapfrog substeps and returns to the host once per
frame for Pillow rendering.

This code is single-process and single-GPU. Requesting four GPUs would waste
three of them. Multi-GPU scaling should be added only for independent
encounters/parameter sweeps or a real distributed solver, with one rank per
GPU and explicit result collation.

## 5. Transfer and playback fallback

Synchronise completed frames from the presentation computer with the transfer
host specified by the event organisers:

```bash
rsync -av USER@TRANSFER_HOST:/path/to/run/ ./runs/leonardo_live/
```

Generate at least one good run of each chosen demonstration before the event
and keep copies on the event PC and removable storage. If a live job queues or
the venue connection fails, label the replay honestly as a run computed
earlier on Leonardo.

## Official references

- [CINECA Leonardo hardware, filesystems, partitions, and current known issues](https://docs.hpc.cineca.it/hpc/leonardo.html)
- [CINECA scheduler and job-submission guide](https://docs.hpc.cineca.it/hpc/hpc_scheduler.html)
- [CINECA Singularity and Leonardo CUDA/container guidance](https://docs.hpc.cineca.it/services/singularity.html)
- [CINECA Python `cineca-ai` environment guidance](https://docs.hpc.cineca.it/hpc/hpc_cineca-ai-hpyc.html)
