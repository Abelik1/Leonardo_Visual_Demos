#!/usr/bin/env bash
set -euo pipefail
DEMO=${1:-reaction_diffusion}
FRAMES=${2:-90}
PROFILE=${3:-leonardo}
ACCOUNT=${LEONARDO_ACCOUNT:-EUHPC_TDEMO_26}
RUNROOT=${LEONARDO_RUNROOT:-$FAST/leonardo_visual_demos}
mkdir -p "$RUNROOT"
sbatch --export=ALL,DEMO="$DEMO",FRAMES="$FRAMES",PROFILE="$PROFILE",RUNROOT="$RUNROOT" --account="$ACCOUNT" slurm/run_demo.sbatch
