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
#SBATCH --array=0-3

DSET_LIST=(CBG_Z18 HT_LIF24 PaviaATN)
NDIM_LIST=(3 2 2)

DATASET=${DSET_LIST[$SLURM_ARRAY_TASK_ID]}
NDIM=${NDIM_LIST[$SLURM_ARRAY_TASK_ID]}

python scripts/run_frc_analysis_on_dataset.py \
    --dataset $DATASET \
    --predictions_subdir predictions_MMSE64 \
    --methods inner_tiling SWITi \
    --ndim $NDIM \
    --output_root /project/careamics/switi/results/$DATASET/predictions_MMSE64/frc/
