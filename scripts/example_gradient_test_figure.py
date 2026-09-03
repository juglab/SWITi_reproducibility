"""Create example gradient-test overlay figures for one image.

The script reads gradient-test reports from
`<output-dir>/<dataset>/<prediction-subdir>/gradient_test/`, loads the matching
prediction or ground-truth image, and saves one significance-overlay PNG per
method and channel listed in `summary.csv`.
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tilartmetrics.gradient_test.plotting import plot_significance_overlay
from tilartmetrics.gradient_test.aggregation import MethodReport
import tifffile

from utils.io_utils import list_files

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
        description="Create example figures to display the gradient test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="dataset name used to find ground-truth data",
    )
    p.add_argument(
        "--prediction-root",
        type=Path,
        required=True,
        help="Root holding {dataset}/{prediction_subdir}/{method}/predictions.npz.",
    )
    p.add_argument(
        "--prediction-subdir",
        default="predictions_MMSE64",
        help="Predictions folder under the dataset (e.g. predictions_MMSE64).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root holding {dataset}/{inputs,targets}/{train,val,test}.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gradient_test"),
        help=(
            "Root where gradient test outputs are written; figures are saved under "
            "{output_dir}/{dataset}/{prediction_subdir}/gradient_test."
        ),
    )
    p.add_argument(
        "--image-name",
        required=True,
        type=str,
        help="Name of the input image to plot the figures for (file name stem).",
    )
    return p.parse_args()


def plot_figure(
    method: Literal["inner_tiling", "GT", "SWITi"],
    img_name: str,
    pred_root: Path,
    data_root: Path,
    grad_test_dir: Path,
    channel: int,
    tile_size: tuple[int, ...],
    overlap: tuple[int, ...],
) -> Figure:
    """Create a gradient-test significance overlay for one method and channel.

    Parameters
    ----------
    method : {"inner_tiling", "GT", "SWITi"}
        Image source to plot.
    img_name : str
        Input image stem to plot.
    pred_root : Path
        Prediction directory containing method subdirectories.
    data_root : Path
        Dataset directory containing input and target TIFF files.
    grad_test_dir : Path
        Directory containing gradient-test reports.
    channel : int
        Channel index to plot.
    tile_size : tuple of int
        Tile size used by the gradient test.
    overlap : tuple of int
        Tile overlap used by the gradient test.

    Returns
    -------
    Figure
        Matplotlib figure containing the overlay.
    """
    img = get_image(
        method=method, img_name=img_name, pred_root=pred_root, data_root=data_root
    )
    if img.ndim == 5:
        z_idx = img.shape[-3] // 2
    else:
        z_idx = None

    report_path = grad_test_dir / f"{method}_gradient_report.json"
    with open(report_path, "r") as f:
        report = MethodReport(**json.load(f))

    fig = plot_significance_overlay(
        report.images[img_name],
        img[0, channel],
        title=method,
        channel=channel,
        tile_size=tile_size,
        overlap=overlap,
        overlay_cmap="Reds",
        overlay_alpha=1,
        gradient_percentile=90,
    )
    fig_title = f"image: {img_name} | channel: {channel}"
    if z_idx is not None:
        fig_title = fig_title + f" | z_idx: {z_idx}"
    fig.suptitle(fig_title, c="w")
    return fig


def get_image(
    method: Literal["inner_tiling", "GT", "SWITi"],
    img_name: str,
    pred_root: Path,
    data_root: Path,
) -> np.ndarray:
    """Load a prediction or ground-truth image.

    Parameters
    ----------
    method : {"inner_tiling", "GT", "SWITi"}
        Image source to load.
    img_name : str
        Input image stem.
    pred_root : Path
        Prediction directory containing method subdirectories.
    data_root : Path
        Dataset directory containing input and target TIFF files.

    Returns
    -------
    np.ndarray
        Loaded image array.
    """
    if method == "GT":
        target_name = get_gt_target_name(img_name, data_root)
        img = tifffile.imread(data_root / "targets" / "test" / f"{target_name}.tif")[
            np.newaxis
        ]
    else:
        predictions = np.load(pred_root / METHODS_TO_SUBDIR[method] / "predictions.npz")
        img = predictions[img_name]

    return img


def get_gt_target_name(img_name: str, data_root: Path) -> str:
    """Return the target image stem matching an input image stem.

    Parameters
    ----------
    img_name : str
        Input image stem.
    data_root : Path
        Dataset directory containing `inputs/test/` and `targets/test/`.

    Returns
    -------
    str
        Matching target image stem.
    """
    input_paths = list_files(data_root, split="test", subset="inputs")
    target_paths = list_files(data_root, split="test", subset="targets")
    target_name = target_paths[
        input_paths.index(data_root / "inputs" / "test" / f"{img_name}.tif")
    ].stem
    return target_name


def read_tile_size_and_overlap(
    pred_dir: Path,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return tile size and overlap from the inner-tiling sidecar.

    Parameters
    ----------
    pred_dir : Path
        Prediction directory containing method subdirectories.

    Returns
    -------
    tuple of tuple of int
        Tile size and overlap.

    Raises
    ------
    FileNotFoundError
        If `inner_tiling/inference_params.json` cannot be found.
    """

    inference_params_path = pred_dir / "inner_tiling" / "inference_params.json"
    try:
        with open(inference_params_path) as f:
            inference_params = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find inference parameters at {inference_params_path}."
        ) from e

    tile_size = tuple(inference_params["tile_size"])
    overlap = tuple(inference_params["overlap"])

    return tile_size, overlap


def main() -> None:
    """Create significance-overlay figures for the requested image.

    Figures are saved in the gradient-test output directory for the selected
    dataset and prediction subdirectory.
    """
    args = parse_args()

    prediction_dir = args.prediction_root / args.dataset / args.prediction_subdir
    data_dir = args.data_root / args.dataset
    gradient_test_dir = (
        args.output_dir / args.dataset / args.prediction_subdir / "gradient_test"
    )

    tile_size, overlap = read_tile_size_and_overlap(prediction_dir)
    summary_df = pd.read_csv(gradient_test_dir / "summary.csv")
    method_channel = summary_df[["method_name", "channel"]]
    for _, (method, channel) in method_channel.iterrows():
        print(
            f"Creating figure for method={method}, image={args.image_name}, "
            f"channel={channel}"
        )
        fig = plot_figure(
            method=method,
            img_name=args.image_name,
            pred_root=prediction_dir,
            data_root=data_dir,
            grad_test_dir=gradient_test_dir,
            channel=channel,
            tile_size=tile_size,
            overlap=overlap,
        )
        fig.savefig(
            gradient_test_dir / f"significance_overlay_{method}_channel{channel}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
