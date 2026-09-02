#!/bin/bash
#SBATCH --job-name=switi_gradient_test
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=federico.carrara@fht.org
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/project/careamics/switi/hpc_outputs/switi_gradient_test/output_%j.log
#SBATCH --mem=64GB
#SBATCH --array=0-2

DATASET_LIST=(CBG_Z18 HT_LIF24 PaviaATN)
DATA_ROOT=/project/careamics/switi/data
PREDICTION_ROOT=/project/careamics/switi/resultslts

DATASET=${DATASET_LIST[$SLURM_ARRAY_TASK_ID]}

uv run python scripts/run_gradient_test_on_dataset.py \
    --dataset "$DATASET" \
    --prediction-root "$PREDICTION_ROOT" \
    --data-root "$DATA_ROOT" \
    --prediction-subdir predictions_MMSE64 \
    --methods inner_tiling SWITi \
    --strip-width 1 \
    --output-dir "$PREDICTION_ROOT"
