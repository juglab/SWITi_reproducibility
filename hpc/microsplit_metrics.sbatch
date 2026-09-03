#!/bin/bash
#SBATCH --job-name=switi_metrics
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=federico.carrara@fht.org
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/project/careamics/switi/hpc_outputs/switi_metrics/output_%j.log
#SBATCH --mem=64GB
#SBATCH --array=0-2

# Compute global metrics (PSNR, LPIPS, MS-SSIM, MicroMS3IM, Pearson) on the
# stitched predictions produced by the inference jobs. Each array task handles
# one dataset and scores both tiling variants. Run configuration (data dir,
# tile size, overlap, mmse count, ckpt) is read from each prediction dir's
# `inference_params.json`, while --data-root points to the renamed Zenodo dataset
# directories. A GPU is requested to accelerate LPIPS.

DATASET_LIST=(CBG_Z18 HT_LIF24 PaviaATN)
DATA_ROOT=/project/careamics/switi-zenodo/data
PREDICTION_ROOT=/project/careamics/switi/results
MMSE_COUNT=64
METHOD_LIST=(inner_tiling SWITi)

DATASET=${DATASET_LIST[$SLURM_ARRAY_TASK_ID]}

for METHOD in "${METHOD_LIST[@]}"; do
    uv run python -u scripts/microsplit_compute_metrics.py \
        --dataset "$DATASET" \
        --prediction-root "$PREDICTION_ROOT" \
        --prediction-subdir "predictions_MMSE$MMSE_COUNT" \
        --method "$METHOD" \
        --data-root "$DATA_ROOT" \
        --split test
done
