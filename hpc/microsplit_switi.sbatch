#!/bin/bash
#SBATCH --job-name=switi_inference
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=federico.carrara@fht.org
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/project/careamics/switi/hpc_outputs/switi_inference/output_%j.log
#SBATCH --mem=64GB
#SBATCH --array=0-2

DSET_LIST=(CBG_Z18 HT_LIF24 PaviaATN)
OVERLAP_LIST=("2 32 32" "32 32" "32 32")
STRIDE_Z_LIST=(1 None None)
DATA_ROOT=/project/careamics/switi-zenodo/data
CHECKPOINT_ROOT=/project/careamics/switi-zenodo/checkpoints
PREDICTION_ROOT=/project/careamics/switi/results

DATASET=${DSET_LIST[$SLURM_ARRAY_TASK_ID]}
OVERLAP=${OVERLAP_LIST[$SLURM_ARRAY_TASK_ID]}
STRIDE_Z=${STRIDE_Z_LIST[$SLURM_ARRAY_TASK_ID]}


ARGS=(
    --dataset "$DATASET"
    --split test
    --data-root "$DATA_ROOT"
    --checkpoint-root "$CHECKPOINT_ROOT"
    --prediction-root "$PREDICTION_ROOT"
    --overlap $OVERLAP
    --mmse-count 64
    --batch-size 128
    --num-workers 3
    --device cuda
)
if [[ "$STRIDE_Z" != "None" ]]; then
    ARGS+=(--stride-z "$STRIDE_Z")
fi

uv run python -u scripts/microsplit_switi_inference.py "${ARGS[@]}"
