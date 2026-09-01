#!/bin/bash
#SBATCH --job-name=visual-bench-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH --time=02:00:00
#SBATCH --partition=dcgp_usr_prod
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
module purge
module load "${LEONARDO_PROFILE_MODULE:-profile/base}"
export LEONARDO_DEMO_CPU_WORKERS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
: "${FAST:?FAST is required for benchmark output}"
PYTHON=${LEONARDO_VENV:+$LEONARDO_VENV/bin/}python3
OUT=${BENCHMARK_ROOT:-$FAST/leonardo_visual_demos/benchmarks}/cpu_${SLURM_JOB_ID}
cd "${SLURM_SUBMIT_DIR}"
srun --cpu-bind=cores "$PYTHON" tools/leonardo_preflight.py --backend cpu --demo crystal --run-root "$OUT"
srun --cpu-bind=cores "$PYTHON" tools/benchmark_demos.py --profile "${BENCH_PROFILE:-benchmark}" \
  --frames "${BENCH_FRAMES:-4}" --backends cpu --all-methods --output "$OUT" --python "$PYTHON"
