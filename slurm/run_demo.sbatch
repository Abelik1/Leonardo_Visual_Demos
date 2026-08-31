#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
DEMO=${DEMO:-reaction_diffusion}
FRAMES=${FRAMES:-90}
PROFILE=${PROFILE:-leonardo}
RUNROOT=${RUNROOT:-$FAST/leonardo_visual_demos}
RUN_DIR="$RUNROOT/${DEMO}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# Adapt these module names to the exact software stack available during the event.
module purge || true
module load profile/base || true
module load python || true
# For CuPy/PyTorch, activate a tested venv/container that contains the matching CUDA build.
if [ -n "${LEONARDO_VENV:-}" ] && [ -f "$LEONARDO_VENV/bin/activate" ]; then source "$LEONARDO_VENV/bin/activate"; fi

export LEONARDO_DEMO_BACKEND=${LEONARDO_DEMO_BACKEND:-auto}
# Avoid oversubscribing the host cores reserved by --cpus-per-task.  The
# hybrid backend uses two of them for concurrent JPEG encoding; libraries that
# use OpenMP/MKL may use the remaining allocation for their own CPU paths.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd "${SLURM_SUBMIT_DIR}"
srun python run_demo.py "$DEMO" --profile "$PROFILE" --frames "$FRAMES" --run-dir "$RUN_DIR"

echo "Run written to $RUN_DIR"
