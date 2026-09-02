"""Run the FRC metric on every image of a dataset, per method.

Experimental driver, argparse-based so it can be launched from a SLURM job. The
FRC counterpart of ``scripts/run_gradient_test_on_dataset.py``: same data layout,
same method->subdir map, same report/summary outputs — swap ``gradient_test`` for
``frc`` and you are done.

Unlike the gradient test, FRC is a *reference* metric: every prediction is scored
against its matching ground truth, so the GT is mandatory (there is no seam-free
"null" source to append). FRC is also 2-D only; a 3-D volume ``(C, D, H, W)`` is
scored per z-slice, each z-plane contributing an extra image ``{name}_z{d:03d}``.

Data layout (see ``playground.ipynb``): for a dataset ``D`` each method stores a
single ``predictions.npz`` whose keys are image names and whose arrays squeeze to
channel-first ``(C, H, W)`` (2-D) or ``(C, D, H, W)`` (3-D); the matching ground
truth lives at ``{data_dir}/{D}/targets/test/{image_name}.tif``. All images of a
dataset must share the same spatial size — FRC pools a per-bin mean curve + 95%
CI across images, which requires a common frequency grid (a clear error is raised
otherwise).

Outputs (under ``{output_dir}``):
- ``{method}_frc_report.json`` — the full per-method report (nested
  image -> channel -> FRC curve), loadable via ``FRCMethodReport.load``.
- ``summary.csv`` — one row per (method, image, channel) from ``to_records()``.
- ``frc_curves_c{channel}.png`` — headline figure per analysed channel: every
  method's mean FRC curve with a shaded 95% CI band.

Example::

    python scripts/run_frc_analysis_on_dataset.py \\
        --dataset PaviaATN --predictions_subdir predictions_MMSE64 \\
        --methods inner_tiling SWITi
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")  # headless: write figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff

from tilartmetrics.frc.aggregation import FRCMethodReport
from tilartmetrics.frc.analysis import run_frc_analysis_dataset
from tilartmetrics.frc.plotting import plot_frc_curves, shared_ylim


METHODS_TO_SUBDIR = {
    "inner_tiling": "inner_tiling",
    "SWITi": "sw_inner_tiling",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Run the FRC metric on all images of a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data location.
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. PaviaATN.")
    p.add_argument(
        "--preds_dir",
        type=Path,
        default=Path("/project/careamics/switi/results"),
        help="Root holding {dataset}/{predictions_subdir}/{method}/predictions.npz.",
    )
    p.add_argument(
        "--data_dir",
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
    # FRC geometry / parameters.
    p.add_argument(
        "--ndim",
        required=True,
        type=int, 
        choices=[2, 3],
        help="Spatial dimensionality; 3-D volumes are scored per z-slice.",
    )
    p.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="Channel indices to analyse (default: all channels).",
    )
    p.add_argument(
        "--step",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Seam interval (px) per method, matching --methods 1:1: the spacing "
            "at which that method lays down seams (inner_tiling: "
            "tile_size - overlap; SWITi: the sliding stride). Draws dashed "
            "verticals on the plot at the seam harmonics k/step (k=1..step//2). "
            "Use 'none' for a seam-free method. Default: no verticals."
        ),
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap images per method for quick trials (default: all).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/frc"),
        help="Reports + summary.csv are written under {output_dir}.",
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


def parse_steps(
    tokens: list[str] | None, methods: list[str]
) -> dict[str, int | None] | None:
    """Map ``--step`` tokens onto ``--methods``, 1:1.

    Each token is either a positive int — the method's seam interval in pixels
    (``tile_size - overlap`` for inner tiling, the sliding stride for SWITi) —
    or ``"none"`` for a seam-free method. Returns ``None`` when the flag was not
    supplied, which disables the harmonic verticals.
    """
    if tokens is None:
        return None
    if len(tokens) != len(methods):
        raise ValueError(
            f"--step has {len(tokens)} entries but --methods has "
            f"{len(methods)}; pass one value per method ('none' is allowed)"
        )
    out: dict[str, int | None] = {}
    for method, token in zip(methods, tokens):
        t = token.strip().lower()
        if t in ("none", ""):
            out[method] = None
            continue
        try:
            value = int(t)
        except ValueError as e:
            raise ValueError(
                f"invalid --step token {token!r} for method {method!r}: "
                "expected a positive int or 'none'"
            ) from e
        if value < 2:
            raise ValueError(
                f"--step must be >= 2 (a step of 1 has no harmonic below "
                f"Nyquist), got {value} for method {method!r}"
            )
        out[method] = value
    return out


def _gt_filename(name: str) -> str:
    """Map a prediction image name to its ground-truth filename.

    The ``predictions.npz`` keys mirror the *input* image names (``input_img_*``),
    while the ground truths are stored as ``target_img_*.tif``; translate the
    ``input`` prefix to ``target`` so the two line up.
    """
    return f"{name.replace('input', 'target', 1)}.tif"


def iter_frc_pairs(
    npz_path: Path,
    target_dir: Path,
    image_names: list[str],
    n_spatial: int,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Lazily yield ``(image_id, prediction, ground_truth)`` 2-D slices.

    Each ``prediction`` / ``ground_truth`` is channel-first ``(C, H, W)``.
    ``.npz`` archives decompress each array only on access, so only one image is
    held in memory at a time. For ``n_spatial == 3`` each volume ``(C, D, H, W)``
    is expanded into ``D`` per-z-slice images with ids ``f"{name}_z{d:03d}"``.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        for name in image_names:
            pred = _ensure_channel_first(data[name], n_spatial)
            gt = _ensure_channel_first(
                tiff.imread(target_dir / _gt_filename(name)), n_spatial
            )
            if n_spatial == 2:
                yield name, pred, gt
            else:  # 3-D: score each z-plane as its own 2-D image.
                depth = pred.shape[1]
                for d in range(depth):
                    yield f"{name}_z{d:03d}", pred[:, d], gt[:, d]


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.preds_dir / args.dataset / args.predictions_subdir
    target_dir = args.data_dir / args.dataset / "targets" / "test"

    # Validate up-front so a bad value fails before the expensive FRC sweep.
    steps = parse_steps(args.step, args.methods)
    if steps is None:
        print(
            "note: --step not given -> no seam-harmonic verticals on the plot. "
            "Pass one value per method, e.g. --step 32 16 (or 'none' for a "
            "seam-free method)."
        )

    # Image names shared across methods (cheap: reads the archive directory).
    first_subdir = METHODS_TO_SUBDIR[args.methods[0]]
    image_names = read_image_names(
        pred_dir / first_subdir / "predictions.npz", args.max_images
    )

    reports: dict[str, FRCMethodReport] = {}
    for method_name in args.methods:
        print(f"\n=== {args.dataset} / {method_name}: {len(image_names)} images ===")
        pairs = iter_frc_pairs(
            pred_dir / METHODS_TO_SUBDIR[method_name] / "predictions.npz",
            target_dir,
            image_names,
            args.ndim,
        )
        report = run_frc_analysis_dataset(
            pairs,
            method_name=method_name,
            dataset=args.dataset,
            channels=args.channels,
            apply_window=True,
        )
        report.save(out_dir / f"{method_name}_frc_report.json")
        reports[method_name] = report

    records = [row for report in reports.values() for row in report.to_records()]
    df = pd.DataFrame.from_records(records)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    # Headline figure: per-method mean FRC curve + 95% CI band (+ dashed seam
    # harmonics k/step where --step gives one), one file per analysed channel
    # (channels are the keys of each method's mean_frc dict).
    report_list = list(reports.values())
    channels_plotted = sorted({c for report in report_list for c in report.mean_frc})
    # One y-range for the whole dataset so the per-channel panels are directly
    # comparable; the title is derived from each report's `dataset` stamp.
    ylim = shared_ylim(report_list, channels_plotted)
    for c in channels_plotted:
        fig = plot_frc_curves(
            report_list,
            steps,
            save_path=out_dir / f"frc_curves_ch{c}.pdf",
            channel=c,
            ylim=ylim,
        )
        plt.close(fig)

    print(
        f"\nSaved {len(reports)} reports + {len(df)}-row summary + "
        f"{len(channels_plotted)} curve plot(s) to {out_dir}"
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
