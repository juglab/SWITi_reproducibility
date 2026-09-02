"""Run the gradient permutation test on every image of a dataset, per method.

Experimental driver, argparse-based so it can be launched from a SLURM job.

Data layout (see ``playground.ipynb``): for a dataset ``D`` each method stores a
single ``predictions.npz`` whose keys are image names and whose arrays squeeze to
channel-first ``(C, H, W)`` (2-D) or ``(C, D, H, W)`` (3-D) — the spatial
dimensionality is inferred from the number of ``--tile_size`` entries; the
matching ground truth lives at ``{data_root}/{D}/targets/test/{image_name}.tif``.
Images within a dataset may differ in size, so we test them one at a time
(``N=1``) and merge the per-image reports into one :class:`MethodReport` per
method.

Outputs (under ``{output_root}/{dataset}``):
- ``{method}_gradient_report.json`` — the full per-method report (nested
  image -> channel -> tiles), loadable via ``MethodReport.load``.
- ``summary.csv`` — one row per (method, image, channel) from ``to_records()``.

Example::

    python scripts/run_gradient_test_on_dataset.py \\
        --dataset PaviaATN --predictions_subdir predictions_MMSE64 \\
        --tile_size 64,64 --overlap 32,32 --statistic js
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

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
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Run the gradient permutation test on all images of a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data location.
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. PaviaATN.")
    p.add_argument(
        "--results_root",
        type=Path,
        default=Path("/project/careamics/switi/results"),
        help="Root holding {dataset}/{predictions_subdir}/{method}/predictions.npz.",
    )
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path("/project/careamics/switi/data"),
        help="Root holding {dataset}/targets/test/{image}.tif ground truths.",
    )
    p.add_argument(
        "--predictions_subdir",
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
        "--no_gt",
        dest="include_gt",
        action="store_false",
        default=True,
        help="Skip the ground truth (by default GT is tested as a seam-free null).",
    )
    # Gradient-test geometry / parameters.
    p.add_argument(
        "--tile_size",
        required=True,
        type=int,
        nargs="+",
        help="TiledPatching tile size per spatial axis.",
    )
    p.add_argument(
        "--overlap",
        required=True,
        type=int,
        nargs="+",
        help="TiledPatching overlap per spatial axis.",
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
        "--n_permutations", type=int, default=1000, help="Permutations per tile."
    )
    p.add_argument("--random_seed", type=int, default=0, help="RNG seed.")
    p.add_argument(
        "--strip_width",
        type=int,
        default=4,
        help="Half-width N of the control strip around each seam.",
    )
    p.add_argument(
        "--block_size",
        type=int,
        default=3,
        help="Contiguous-block size B for the permutation engine.",
    )
    p.add_argument(
        "--num_bins_per_tile",
        type=int,
        default=32,
        help="Histogram bins for binned statistics (KL, JS).",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap images per method for quick trials (default: all).",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/gradient_test"),
        help="Reports + summary.csv are written under {output_root}/{dataset}.",
    )
    return p.parse_args()


def _ensure_channel_first(arr: np.ndarray, n_spatial: int) -> np.ndarray:
    """Squeeze to channel-first ``(C, *spatial)`` for the given spatial ndim.

    ``n_spatial`` is 2 for 2-D ``(C, H, W)`` or 3 for 3-D ``(C, D, H, W)``. 
    A bare spatial array with no channel axis (``(H, W)`` / ``(D, H, W)``) is
    promoted to a single channel.
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
    """Return the image names in a ``predictions.npz`` (reads the archive index,
    not the arrays); optionally capped to the first ``max_images``."""
    names = list(np.load(npz_path, allow_pickle=True).files)
    return names if max_images is None else names[:max_images]


def iter_prediction_images(
    npz_path: Path, image_names: list[str], n_spatial: int
) -> Iterator[tuple[str, np.ndarray]]:
    """Lazily yield ``(name, (C, *spatial))`` from a ``predictions.npz``.

    ``.npz`` archives decompress each array only on access, so this keeps just
    one image in memory at a time. ``n_spatial`` (2 or 3) selects the expected
    channel-first layout ``(C, H, W)`` / ``(C, D, H, W)``.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        for name in image_names:
            yield name, _ensure_channel_first(data[name], n_spatial)


def _gt_filename(name: str) -> str:
    """Map a prediction image name to its ground-truth filename.

    The ``predictions.npz`` keys mirror the *input* image names (``input_img_*``),
    while the ground truths are stored as ``target_img_*.tif``; translate the
    ``input`` prefix to ``target`` so the two line up.
    """
    return f"{name.replace('input', 'target', 1)}.tif"


def iter_gt_images(
    target_dir: Path, image_names: list[str], n_spatial: int
) -> Iterator[tuple[str, np.ndarray]]:
    """Lazily yield ``(name, (C, *spatial))`` ground truths, one file at a time.

    ``n_spatial`` (2 or 3) selects the expected channel-first layout.
    """
    for name in image_names:
        yield name, _ensure_channel_first(
            tiff.imread(target_dir / _gt_filename(name)), n_spatial
        )


def main() -> None:
    args = parse_args()
    out_dir = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_root = args.results_root / args.dataset / args.predictions_subdir

    n_spatial = len(args.tile_size)
    if len(args.overlap) != n_spatial:
        raise ValueError(
            f"tile_size has {n_spatial} entries but overlap has "
            f"{len(args.overlap)}; both must list one value per spatial axis"
        )

    cfg = GradientTestConfig(
        tile_size=list(args.tile_size),
        overlap=list(args.overlap),
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
            )
        )
        for name  in args.methods
    ]
    if args.include_gt:
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
