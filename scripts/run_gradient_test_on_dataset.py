"""Run the gradient permutation test for one dataset.

Predictions are read from
`<prediction-root>/<dataset>/<prediction-subdir>/<method>/predictions.npz`.
The matching ground truth is read from `<data-root>/<dataset>/targets/test/`
when ground truth is included. Reports are written to
`<output-dir>/<dataset>/<prediction-subdir>/gradient_test/`. Note that the gradient test
is reference free so the ground truth is not required. The gradient test can be run on
the ground truth to see the baseline rejection rate and ensure that the test is 
calibrated (for alpha=0.05 we should expect a baseline rejection rate of 5%).

Each selected method produces a `{method}_gradient_report.json` file. The script
also writes `summary.csv` and `gradient_test_config.json` in the same output
directory.

Example
-------

    uv run python scripts/run_gradient_test_on_dataset.py \\
        --dataset PaviaATN --prediction-root /path/to/results \\
        --data-root /path/to/data --prediction-subdir predictions_MMSE64 \\
        --tile-size 64 64 --overlap 32 32 --statistic js
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from tilartmetrics.config import GradientTestConfig
from tilartmetrics.gradient_test.aggregation import MethodReport
from tilartmetrics.gradient_test.analysis import run_gradient_analysis_dataset

METHODS_TO_SUBDIR = {
    "inner_tiling": "inner_tiling",
    "SWITi": "sw_inner_tiling",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    p = argparse.ArgumentParser(
        description="Run the gradient permutation test on all images of a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data location.
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. PaviaATN.")
    p.add_argument(
        "--prediction-root",
        type=Path,
        required=True,
        help="Root holding {dataset}/{prediction_subdir}/{method}/predictions.npz.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root holding {dataset}/targets/test/{image}.tif ground truths.",
    )
    p.add_argument(
        "--prediction-subdir",
        default="predictions_MMSE64",
        help="Predictions folder under the dataset (e.g. predictions_MMSE64).",
    )
    p.add_argument(
        "--methods",
        required=True,
        type=str,
        nargs="+",
        choices=["inner_tiling", "SWITi"],
        help="Space-separated list of method names.",
    )
    p.add_argument(
        "--no-gt",
        dest="include_gt",
        action="store_false",
        default=True,
        help="Skip the ground truth (by default GT is tested as a seam-free null).",
    )
    # Gradient-test geometry / parameters.
    p.add_argument(
        "--tile-size",
        type=int,
        nargs="+",
        default=None,
        help=(
            "TiledPatching tile size per spatial axis. If None will attempt to read it "
            "from the inner tiling inference_params.json."
        ),
    )
    p.add_argument(
        "--overlap",
        type=int,
        nargs="+",
        default=None,
        help=(
            "TiledPatching overlap per spatial axis. If None will attempt to read it "
            "from the inner tiling inference_params.json."
        ),
    )
    p.add_argument(
        "--statistic",
        default="js",
        choices=["kl", "js", "ks", "wasserstein", "mean_abs_ratio"],
        help="Two-sample discrepancy statistic.",
    )
    p.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="Channel indices to test (default: all channels).",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="Rejection threshold.")
    p.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Permutations per tile.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="RNG seed.",
    )
    p.add_argument(
        "--strip-width",
        type=int,
        default=4,
        help="Half-width N of the control strip around each seam.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=3,
        help="Contiguous-block size B for the permutation engine.",
    )
    p.add_argument(
        "--num-bins-per-tile",
        type=int,
        default=32,
        help="Histogram bins for binned statistics (KL, JS).",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap images per method for quick trials (default: all).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gradient_test"),
        help="Reports + summary.csv are written under {output_dir}/{dataset}/{prediction_subdir}/gradient_test.",
    )
    return p.parse_args()


def _ensure_channel_first(arr: np.ndarray, n_spatial: int) -> np.ndarray:
    """Return an array with channel-first axes.

    Parameters
    ----------
    arr : np.ndarray
        Image array with optional singleton axes.
    n_spatial : int
        Number of spatial axes, either 2 for `(Y, X)` or 3 for `(Z, Y, X)`.

    Returns
    -------
    np.ndarray
        Image array with shape `(C, *spatial)`.

    Raises
    ------
    ValueError
        If the squeezed array cannot be interpreted as channel-first.
    """
    arr = np.asarray(arr).squeeze()
    if arr.ndim == n_spatial:
        arr = arr[np.newaxis, ...]
    if arr.ndim != n_spatial + 1:
        raise ValueError(
            f"expected {n_spatial + 1}-D channel-first array after squeeze "
            f"(n_spatial={n_spatial}), got shape {arr.shape}"
        )
    return arr


def read_image_names(npz_path: Path, max_images: int | None) -> list[str]:
    """Return image names stored in a prediction archive.

    Parameters
    ----------
    npz_path : Path
        Path to a `predictions.npz` archive.
    max_images : int or None
        Maximum number of names to return. If `None`, all names are returned.

    Returns
    -------
    list of str
        Prediction archive keys in archive order.
    """
    names = list(np.load(npz_path, allow_pickle=True).files)
    return names if max_images is None else names[:max_images]


def iter_prediction_images(
    npz_path: Path, image_names: list[str], n_spatial: int
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield prediction images from a prediction archive.

    Parameters
    ----------
    npz_path : Path
        Path to a `predictions.npz` archive.
    image_names : list of str
        Archive keys to read.
    n_spatial : int
        Number of spatial axes in each image.

    Yields
    ------
    tuple of str and np.ndarray
        Image name and channel-first prediction array.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        for name in image_names:
            yield name, _ensure_channel_first(data[name], n_spatial)


def _gt_filename(name: str) -> str:
    """Map a prediction image name to its ground-truth filename.

    Parameters
    ----------
    name : str
        Prediction image name, usually matching an input file stem.

    Returns
    -------
    str
        Ground-truth TIFF filename.
    """
    return f"{name.replace('input', 'target', 1)}.tif"


def iter_gt_images(
    target_dir: Path, image_names: list[str], n_spatial: int
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ground-truth images matching prediction names.

    Parameters
    ----------
    target_dir : Path
        Directory containing target TIFF files.
    image_names : list of str
        Prediction names to match.
    n_spatial : int
        Number of spatial axes in each image.

    Yields
    ------
    tuple of str and np.ndarray
        Prediction name and matching channel-first ground-truth array.
    """
    for name in image_names:
        yield (
            name,
            _ensure_channel_first(
                tiff.imread(target_dir / _gt_filename(name)), n_spatial
            ),
        )


def read_tile_size_and_overlap(
    pred_dir: Path, tile_size: tuple[int, ...] | None, overlap: tuple[int, ...] | None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return tile size and overlap for gradient-test geometry.

    Parameters
    ----------
    pred_dir : Path
        Prediction directory containing method subdirectories.
    tile_size : tuple[int, ...] | None
        Tile size supplied by the caller. If `None`, it is read from
        `inner_tiling/inference_params.json`.
    overlap : tuple[int, ...] | None
        Tile overlap supplied by the caller. If `None`, it is read from
        `inner_tiling/inference_params.json`.

    Returns
    -------
    tuple of tuple of int
        Tile size and overlap.

    Raises
    ------
    FileNotFoundError
        If `inner_tiling/inference_params.json` is needed and cannot be found.
    """
    if tile_size is not None and overlap is not None:
        return tile_size, overlap

    inference_params_path = pred_dir / "inner_tiling" / "inference_params.json"
    try:
        with open(inference_params_path) as f:
            inference_params = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find inference parameters at {inference_params_path}."
        ) from e

    if tile_size is None:
        tile_size = tuple(inference_params["tile_size"])
    if overlap is None:
        overlap = tuple(inference_params["overlap"])

    return tile_size, overlap


def main() -> None:
    """Run gradient tests for the requested dataset.

    Reports, summaries, and the saved test configuration are written under the
    selected output directory.
    """
    args = parse_args()
    out_dir = args.output_dir / args.dataset / args.prediction_subdir / "gradient_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_root = args.prediction_root / args.dataset / args.prediction_subdir

    tile_size, overlap = read_tile_size_and_overlap(
        pred_root, args.tile_size, args.overlap
    )
    n_spatial = len(tile_size)
    if len(overlap) != n_spatial:
        raise ValueError(
            f"tile_size has {n_spatial} entries but overlap has "
            f"{len(overlap)}; both must list one value per spatial axis"
        )

    cfg = GradientTestConfig(
        tile_size=list(tile_size),
        overlap=list(overlap),
        statistic=args.statistic,
        strip_width=args.strip_width,
        block_size=args.block_size,
        n_permutations=args.n_permutations,
        alpha=args.alpha,
        num_bins_per_tile=args.num_bins_per_tile,
        random_seed=args.random_seed,
        normalize_per_axis=True,
        balance_axis_counts=True,
        channels=args.channels,
    )
    (out_dir / "gradient_test_config.json").write_text(cfg.model_dump_json(indent=2))

    # Image names shared across methods/GT (cheap: reads the archive directory).
    first_subdir = METHODS_TO_SUBDIR[args.methods[0]]
    image_names = read_image_names(
        pred_root / first_subdir / "predictions.npz", args.max_images
    )

    # (method_name, lazy image iterator) sources, GT appended as a null reference.
    sources: list[tuple[str, Iterator[tuple[str, np.ndarray]]]] = [
        (
            name,
            iter_prediction_images(
                pred_root / METHODS_TO_SUBDIR[name] / "predictions.npz",
                image_names,
                n_spatial,
            ),
        )
        for name in args.methods
    ]
    if args.include_gt:
        if args.data_root is None:
            raise ValueError("--data-root is required unless --no-gt is passed")
        target_dir = args.data_root / args.dataset / "targets" / "test"
        sources.append(("GT", iter_gt_images(target_dir, image_names, n_spatial)))

    reports: dict[str, MethodReport] = {}
    for method_name, image_iter in sources:
        print(f"\n=== {args.dataset} / {method_name}: {len(image_names)} images ===")
        report = run_gradient_analysis_dataset(
            image_iter,
            tile_size=cfg.tile_size,
            overlap=cfg.overlap,
            method_name=method_name,
            dataset=args.dataset,
            channels=cfg.channels,
            statistic=cfg.statistic,
            strip_width=cfg.strip_width,
            block_size=cfg.block_size,
            n_permutations=cfg.n_permutations,
            alpha=cfg.alpha,
            num_bins_per_tile=cfg.num_bins_per_tile,
            random_seed=cfg.random_seed,
            normalize_per_axis=cfg.normalize_per_axis,
            balance_axis_counts=cfg.balance_axis_counts,
        )
        report.save(out_dir / f"{method_name}_gradient_report.json")
        reports[method_name] = report

    records = [row for report in reports.values() for row in report.to_records()]
    df = pd.DataFrame.from_records(records)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaved {len(reports)} reports + {len(df)}-row summary to {out_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
