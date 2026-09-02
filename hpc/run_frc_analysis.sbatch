#!/bin/bash
#SBATCH --job-name=switi_frc
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=federico.carrara@fht.org
#SBATCH --partition=cpuq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/project/careamics/switi/hpc_outputs/switi_frc/output_%j.log
#SBATCH --mem=64GB
#SBATCH --array=0-2

DATASET_LIST=(CBG_Z18 HT_LIF24 PaviaATN)
NDIM_LIST=(3 2 2)
DATA_ROOT=/project/careamics/switi/data
PREDICTION_ROOT=/project/careamics/switi/results

DATASET=${DATASET_LIST[$SLURM_ARRAY_TASK_ID]}
NDIM=${NDIM_LIST[$SLURM_ARRAY_TASK_ID]}

uv run python scripts/run_frc_analysis_on_dataset.py \
    --dataset "$DATASET" \
    --prediction-root "$PREDICTION_ROOT" \
    --data-root "$DATA_ROOT" \
    --prediction-subdir predictions_MMSE64 \
    --methods inner_tiling SWITi \
    --ndim "$NDIM" \
    --output-dir "$PREDICTION_ROOT"
