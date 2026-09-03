# SWITi Reproducibility

This repository reproduces the experiments for [SWITi: Quantifying and Reducing Tiling Artifacts with Sliding Window Inner Tiling](https://arxiv.org/abs/2607.18990). SWITi is a test-time procedure for posterior image-splitting models that averages shifted tiled predictions to reduce seam artifacts without increasing the number of forward passes required for an MMSE estimate. The paper also introduces reference-free tiling-artifact metrics based on gradient-distribution tests.

The data and MicroSplit model weights are downloaded from Zenodo:
- Dataset: https://zenodo.org/records/22213583
- Model weights: https://zenodo.org/records/22214478.

SWITi has been implemented in [CAREamics](https://github.com/CAREamics/careamics), an image restoration deep-learning library. (Currently in the feature branch `mc/feat/switi-merge`).

The gradient test and Fourier Ring Correlation (FRC) metrics are implemented in the repository [juglab/TilArtMetrics](https://github.com/juglab/TilArtMetrics/tree/main).

## Running the Reproduction

We recommend using `uv` to run the scripts from the repository root with `uv run`. 
See the [`uv` documentation](https://docs.astral.sh/uv/).

1. Download data and checkpoints.

    ```bash
    uv run python scripts/download.py --output-dir switi
    ```

    This creates `switi/data` and `switi/checkpoints`.

2. Run baseline inner-tiling inference.

    ```bash
    uv run python scripts/microsplit_inner_tiling_inference.py \
        --dataset HT_LIF24 \
        --data-root switi/data \
        --checkpoint-root switi/checkpoints \
        --prediction-root switi/results \
        --overlap 32 32 \
        --mmse-count 64
    ```

    This writes stitched predictions and `inference_params.json` under `inner_tiling`.

3. Run SWITi inference.

    ```bash
    uv run python scripts/microsplit_switi_inference.py \
        --dataset HT_LIF24 \
        --data-root switi/data \
        --checkpoint-root switi/checkpoints \
        --prediction-root switi/results \
        --overlap 32 32 \
        --mmse-count 64
    ```

    For 3D datasets, pass a 3-axis overlap and `--stride-z`, for example `--overlap 2 32 32 --stride-z 1`.

4. Compute global image-quality metrics.

    ```bash
    uv run python scripts/microsplit_compute_metrics.py \
        --dataset HT_LIF24 \
        --prediction-root switi/results \
        --prediction-subdir predictions_MMSE64 \
        --method inner_tiling \
        --data-root switi/data

    uv run python scripts/microsplit_compute_metrics.py \
        --dataset HT_LIF24 \
        --prediction-root switi/results \
        --prediction-subdir predictions_MMSE64 \
        --method SWITi \
        --data-root switi/data
    ```

    This writes `metrics.json` and `metrics_per_image.json` beside each method's predictions.

5. Run FRC analysis.

    ```bash
    uv run python scripts/run_frc_analysis_on_dataset.py \
        --dataset HT_LIF24 \
        --prediction-root switi/results \
        --prediction-subdir predictions_MMSE64 \
        --methods inner_tiling SWITi \
        --data-root switi/data \
        --ndim 2 \
        --output-dir switi/results
    ```

    This writes FRC reports, `summary.csv`, and per-channel FRC curve figures under `frc`.

6. Run the gradient permutation test.

    ```bash
    uv run python scripts/run_gradient_test_on_dataset.py \
        --dataset HT_LIF24 \
        --prediction-root switi/results \
        --prediction-subdir predictions_MMSE64 \
        --methods inner_tiling SWITi \
        --data-root switi/data \
        --strip-width 1 \
        --output-dir switi/results
    ```

    This writes reports, `summary.csv`, and `gradient_test_config.json` under `gradient_test`.


7. Create example gradient-test overlay figures.

    ```bash
    uv run python scripts/example_gradient_test_figure.py \
        --dataset HT_LIF24 \
        --prediction-root switi/results \
        --prediction-subdir predictions_MMSE64 \
        --data-root switi/data \
        --output-dir switi/results \
        --image-name <input-image-stem>
    ```

    The figure script reads the gradient-test reports and writes significance-overlay PNGs into the same `gradient_test` directory.

## Expected Directory Structure

After the scripts above have run, the example directory should contain:

```text
switi/
├── data/
│   └── HT_LIF24/
│       ├── inputs/
│       │   ├── train/*.tif
│       │   ├── val/*.tif
│       │   └── test/*.tif
│       ├── targets/
│       │   ├── train/*.tif
│       │   ├── val/*.tif
│       │   └── test/*.tif
│       └── stats.json
├── checkpoints/
│   └── HT_LIF24/
│       ├── BaselineVAECL_best.ckpt
│       └── config.yaml
└── results/
    └── HT_LIF24/
        └── predictions_MMSE64/
            ├── inner_tiling/
            │   ├── predictions.npz
            │   ├── inference_params.json
            │   ├── metrics.json
            │   └── metrics_per_image.json
            ├── sw_inner_tiling/
            │   ├── predictions.npz
            │   ├── inference_params.json
            │   ├── metrics.json
            │   └── metrics_per_image.json
            ├── gradient_test/
            │   ├── gradient_test_config.json
            │   ├── inner_tiling_gradient_report.json
            │   ├── SWITi_gradient_report.json
            │   ├── GT_gradient_report.json
            │   ├── summary.csv
            │   └── significance_overlay_<method>_channel<channel>.png
            └── frc/
                ├── inner_tiling_frc_report.json
                ├── SWITi_frc_report.json
                ├── summary.csv
                └── frc_curves_ch<channel>.pdf
```

- `*.tif`: input and target microscopy images for each split.
- `stats.json`: cached per-channel normalization statistics computed from the training data.
- `BaselineVAECL_best.ckpt`: pretrained MicroSplit checkpoint.
- `config.yaml`: training configuration used to reconstruct the model and prediction data configuration.
- `predictions.npz`: stitched predictions keyed by input image stem.
- `inference_params.json`: tile geometry, MMSE count, checkpoint path, and data path used for inference.
- `metrics.json`: dataset-average global image-quality metrics.
- `metrics_per_image.json`: global image-quality metrics for each predicted image.
- `gradient_test_config.json`: parameters used for the gradient permutation test.
- `<method>_gradient_report.json`: per-method gradient-test report.
- `GT_gradient_report.json`: gradient-test report for the ground truth reference, when included.
- `significance_overlay_<method>_channel<channel>.png`: example gradient-test overlay figures.
- `<method>_frc_report.json`: per-method FRC report.
- `frc_curves_ch<channel>.pdf`: FRC summary figure for one channel.

## HPC Scripts

The `hpc` directory contains SLURM wrappers for the same workflow:

```text
hpc/microsplit_inner_tiling.sbatch
hpc/microsplit_switi.sbatch
hpc/microsplit_metrics.sbatch
hpc/run_gradient_test.sbatch
hpc/run_frc_analysis.sbatch
```
