"""Run FRC analysis for one dataset.

Predictions are read from
`<prediction-root>/<dataset>/<prediction-subdir>/<method>/predictions.npz`.
Ground truth is read from `<data-root>/<dataset>/targets/test/`. Reports are
written to `<output-dir>/<dataset>/<prediction-subdir>/frc/`.

FRC is a *reference* metric: every prediction is scored against its matching ground 
truth. Predictions are matched to ground truth by the sorted input and target file
order.

Each selected method produces a `{method}_frc_report.json` file. The script also
writes `summary.csv` and one `frc_curves_ch{channel}.pdf` figure per analysed
channel. For 3D data, each z-plane is scored as a separate 2D image.

Example
-------

    uv run python scripts/run_frc_analysis_on_dataset.py \\
        --dataset PaviaATN --prediction-root /path/to/results \\
        --data-root /path/to/data --prediction-subdir predictions_MMSE64 \\
        --methods inner_tiling SWITi
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

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
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    p = argparse.ArgumentParser(
        description="Run the FRC metric on all images of a dataset.",
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
        required=True,
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
        "--max-images",
        type=int,
        default=None,
        help="Cap images per method for quick trials (default: all).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Reports + summary.csv are written under {output_dir}/{dataset}/{prediction_subdir}/frc.",
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


def parse_steps(
    tokens: list[str] | None, methods: list[str]
) -> dict[str, int | None] | None:
    """Map seam-step tokens to method names.

    Parameters
    ----------
    tokens : list of str or None
        Step values from `--step`. Use positive integer strings for methods with
        seams and `none` for seam-free methods.
    methods : list of str
        Method names from `--methods`.

    Returns
    -------
    dict of str to int or None, or None
        Per-method seam steps, or `None` when no steps were supplied.

    Raises
    ------
    ValueError
        If the number of step values does not match the number of methods or a
        step value is invalid.
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


def iter_frc_pairs(
    npz_path: Path,
    target_dir: Path,
    image_names: list[str],
    n_spatial: int,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield prediction and ground-truth image pairs for FRC.

    Parameters
    ----------
    npz_path : Path
        Path to a `predictions.npz` archive.
    target_dir : Path
        Directory containing target TIFF files.
    image_names : list of str
        Prediction names to match.
    n_spatial : int
        Number of spatial axes in each source image.

    Yields
    ------
    tuple of str, np.ndarray, and np.ndarray
        Image identifier, prediction, and ground truth. Returned arrays have
        shape `(C, Y, X)`.
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
    """Run FRC analysis for the requested dataset.

    Reports, summaries, and curve figures are written under the selected output
    directory.
    """
    args = parse_args()
    out_dir = args.output_dir / args.dataset / args.prediction_subdir / "frc"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.prediction_root / args.dataset / args.prediction_subdir
    target_dir = args.data_root / args.dataset / "targets" / "test"

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
