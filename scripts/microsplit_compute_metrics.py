"""Compute global metrics on stitched MicroSplit predictions.

The script reads `<prediction-root>/<dataset>/<prediction-subdir>/<method>/`
for a `predictions.npz` file and its `inference_params.json` sidecar, loads the
matching target TIFFs from `<data-root>/<dataset>/targets/<split>/`, and writes
`metrics.json` and `metrics_per_image.json` beside the predictions.

The computed channel-wise global metrics can be selected from (PSNR, LPIPS, MS-SSIM, 
MicroMS3IM, Pearson).

Predictions are matched to ground truth by the sorted input and target file
order. Prediction arrays are expected as `(S, C, [Z], Y, X)`, with missing
leading sample or channel axes accepted when unambiguous.

When a file holds several frames (`S > 1`) each frame is scored as a separate image.

Run from the repo root:

    uv run python scripts/microsplit_compute_metrics.py \\
        --dataset HT_LIF24 --prediction-root /path/to/results \\
        --data-root /path/to/data --prediction-subdir predictions_MMSE64 \\
        --method SWITi
    uv run python scripts/microsplit_compute_metrics.py \\
        --dataset CBG_Z18 --prediction-root /path/to/results --data-root /path/to/data \\
        --prediction-subdir predictions_MMSE64 --method inner_tiling \\
        --metrics PSNR MSSIM Pearson
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from utils.io_utils import list_files
from utils.metrics_utils import compute_unmixing_metrics, log_metrics
from utils.stats import _load_canonical

METHODS_TO_SUBDIR = {
    "inner_tiling": "inner_tiling",
    "SWITi": "sw_inner_tiling",
}


def _ensure_canonical(arr: NDArray, *, is_3d: bool) -> NDArray:
    """Return an array with shape `(S, C, [Z], Y, X)`.

    Parameters
    ----------
    arr : NDArray
        Prediction array with spatial, channel-spatial, or sample-channel-spatial
        axes.
    is_3d : bool
        Whether the spatial axes are `(Z, Y, X)`.

    Returns
    -------
    NDArray
        Prediction array with explicit sample and channel axes.

    Raises
    ------
    ValueError
        If the number of dimensions is not compatible with the requested data
        dimensionality.
    """
    spatial = 3 if is_3d else 2
    target_ndim = spatial + 2  # S, C, plus spatial
    extra = target_ndim - arr.ndim
    if extra < 0 or extra > 2:
        raise ValueError(
            f"Unexpected prediction ndim {arr.ndim} for "
            f"{'3D' if is_3d else '2D'} data: expected one of "
            f"{{{spatial}, {spatial + 1}, {spatial + 2}}}."
        )
    for _ in range(extra):
        arr = arr[np.newaxis]
    return arr


def _center_crop_to_match(pred: NDArray, gt: NDArray) -> tuple[NDArray, NDArray]:
    """Center-crop prediction and ground truth to a common spatial shape.

    Parameters
    ----------
    pred : NDArray
        Prediction image with shape `(C, [Z], Y, X)`.
    gt : NDArray
        Ground-truth image with shape `(C, [Z], Y, X)`.

    Returns
    -------
    tuple of NDArray
        Cropped prediction and ground truth with matching spatial shape.
    """
    if pred.shape == gt.shape:
        return pred, gt
    pred_slices: list[slice] = [slice(None)]
    gt_slices: list[slice] = [slice(None)]
    for dp, dg in zip(pred.shape[1:], gt.shape[1:], strict=True):
        d = min(dp, dg)
        pred_slices.append(slice((dp - d) // 2, (dp - d) // 2 + d))
        gt_slices.append(slice((dg - d) // 2, (dg - d) // 2 + d))
    return pred[tuple(pred_slices)], gt[tuple(gt_slices)]


def _load_pred_gt_pairs(
    predictions_path: Path,
    data_dir: Path,
    split: str,
    *,
    is_3d: bool,
) -> tuple[list[NDArray], list[NDArray], list[str]]:
    """Load predictions and matching GT targets as parallel per-image lists.

    Parameters
    ----------
    predictions_path : Path
        Path to a `predictions.npz` archive.
    data_dir : Path
        Dataset directory containing `inputs/<split>/` and `targets/<split>/`.
    split : str
        Dataset split to load.
    is_3d : bool
        Whether images have `(Z, Y, X)` spatial axes.

    Returns
    -------
    tuple of list
        Prediction images, ground-truth images, and image identifiers. Each image
        has shape `(C, [Z], Y, X)`. Multi-sample files are expanded to one image
        per sample with `<stem>__s{n}` identifiers.

    Raises
    ------
    ValueError
        If inputs and targets cannot be matched or have incompatible shapes.
    KeyError
        If a prediction is missing from the archive.
    """
    npz = np.load(predictions_path)
    npz_keys = set(npz.keys())

    input_files = list_files(data_dir, split, "inputs")
    target_files = list_files(data_dir, split, "targets")
    if len(input_files) != len(target_files):
        raise ValueError(
            f"Input/target count mismatch for split {split!r}: "
            f"{len(input_files)} inputs vs {len(target_files)} targets."
        )

    pred_imgs: list[NDArray] = []
    gt_imgs: list[NDArray] = []
    img_fnames: list[str] = []
    for i, (inp, tgt) in enumerate(zip(input_files, target_files, strict=True)):
        # Predictions are keyed by input stem; fall back to the array-source key.
        key = inp.stem if inp.stem in npz_keys else f"pred_{i:04d}"
        if key not in npz_keys:
            raise KeyError(
                f"No prediction for input {inp.name!r} (tried keys "
                f"{inp.stem!r} and pred_{i:04d}) in {predictions_path}."
            )
        pred = _ensure_canonical(npz[key], is_3d=is_3d).astype(np.float32)
        gt = _load_canonical(tgt, is_3d=is_3d).astype(np.float32)

        if pred.shape[1] != gt.shape[1]:
            raise ValueError(
                f"Channel mismatch for {inp.stem!r}: prediction has "
                f"{pred.shape[1]} channels, target has {gt.shape[1]}."
            )
        if pred.shape[0] != gt.shape[0]:
            raise ValueError(
                f"Sample-count (S) mismatch for {inp.stem!r}: prediction S="
                f"{pred.shape[0]}, target S={gt.shape[0]}."
            )

        n_samples = pred.shape[0]
        for s in range(n_samples):
            p_img, g_img = _center_crop_to_match(pred[s], gt[s])
            pred_imgs.append(p_img)
            gt_imgs.append(g_img)
            img_fnames.append(inp.stem if n_samples == 1 else f"{inp.stem}__s{s}")

    npz.close()
    return pred_imgs, gt_imgs, img_fnames


def main(args: argparse.Namespace) -> Path:
    """Compute and log metrics for one prediction directory.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    Path
        Directory where metric JSON files were written.
    """
    pred_dir = (
        args.prediction_root
        / args.dataset
        / args.prediction_subdir
        / METHODS_TO_SUBDIR[args.method]
    )
    predictions_path = pred_dir / args.predictions_filename
    params_path = pred_dir / args.params_filename
    if not predictions_path.is_file():
        raise SystemExit(f"Predictions file not found: {predictions_path}")
    if not params_path.is_file():
        raise SystemExit(f"Inference-params file not found: {params_path}")

    params = json.loads(params_path.read_text())
    tile_size = params["tile_size"]
    is_3d = len(tile_size) == 3  # [Z, Y, X] -> 3D, [Y, X] -> 2D
    data_dir = args.data_root / args.dataset

    print(f"Loading predictions from {predictions_path}")
    print(f"Loading GT targets from  {data_dir / 'targets' / args.split}")
    pred_imgs, gt_imgs, img_fnames = _load_pred_gt_pairs(
        predictions_path, data_dir, args.split, is_3d=is_3d
    )
    print(f"matched {len(pred_imgs)} image(s), shape per image {pred_imgs[0].shape}")

    metrics_avg, metrics_per_img = compute_unmixing_metrics(
        pred_imgs=pred_imgs,
        gt_imgs=gt_imgs,
        metrics=args.metrics,
        img_fnames=img_fnames,
    )

    # Log both dataset-average and per-image metrics next to the predictions,
    # filling provenance fields straight from inference_params.json.
    ckpt_path = params.get("ckpt_path")
    common = dict(
        log_dir=pred_dir,
        data_path=data_dir / "targets" / args.split,
        mmse_count=params.get("mmse_count"),
        ckpt_name=Path(ckpt_path).name if ckpt_path else None,
        tile_size=tile_size,
        tile_overlap=params.get("overlap"),
    )
    log_metrics(metrics_avg, per_image=False, **common)
    log_metrics(metrics_per_img, per_image=True, **common)
    return pred_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse. If `None`, arguments are read from the command line.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    p = argparse.ArgumentParser(
        prog="microsplit-compute-metrics",
        description=(
            "Compute channel-wise global metrics (PSNR, LPIPS, MS-SSIM, "
            "MicroMS3IM, Pearson) on stitched MicroSplit predictions vs. GT "
            "targets, writing metrics.json / metrics_per_image.json next to the "
            "predictions. Run configuration is read from inference_params.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="dataset name used to find ground-truth data",
    )
    p.add_argument(
        "--prediction-root",
        required=True,
        type=Path,
        help="root holding {dataset}/{prediction-subdir}/{method}/predictions.npz",
    )
    p.add_argument(
        "--prediction-subdir",
        default="predictions_MMSE64",
        help="prediction folder under the dataset (e.g. predictions_MMSE64)",
    )
    p.add_argument(
        "--method",
        required=True,
        choices=["inner_tiling", "SWITi"],
        help="prediction method to score",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "root holding {dataset}/{inputs,targets}/{train,val,test}"
        ),
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="which on-disk split was predicted",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["PSNR", "LPIPS", "MSSIM", "MicroMS3IM", "Pearson"],
        choices=["PSNR", "LPIPS", "MSSIM", "MicroMS3IM", "Pearson"],
        help="metrics to compute (MicroMS3IM is auto-skipped for 3D data)",
    )
    p.add_argument(
        "--predictions-filename",
        default="predictions.npz",
        help="NPZ filename inside the prediction directory",
    )
    p.add_argument(
        "--params-filename",
        default="inference_params.json",
        help="inference-params JSON filename inside the prediction directory",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
