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
DATA_ROOT=/project/careamics/switi/data
CHECKPOINT_ROOT=/project/careamics/switi/ckpts
PREDICTION_ROOT=/project/careamics/switi/results

DATASET=${DSET_LIST[$SLURM_ARRAY_TASK_ID]}
OVERLAP=${OVERLAP_LIST[$SLURM_ARRAY_TASK_ID]}

uv run python -u scripts/microsplit_inner_tiling_inference.py \
    --dataset "$DATASET" \
    --split test \
    --data-root "$DATA_ROOT" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --prediction-root "$PREDICTION_ROOT" \
    --overlap $OVERLAP \
    --mmse-count 64 \
    --batch-size 128 \
    --num-workers 3 \
    --device cuda
