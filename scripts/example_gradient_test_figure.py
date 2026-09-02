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
    input_paths = list_files(data_root, split="test", subset="inputs")
    target_paths = list_files(data_root, split="test", subset="targets")
    target_name = target_paths[
        input_paths.index(data_root / "inputs" / "test" / f"{img_name}.tif")
    ].stem
    return target_name


def read_tile_size_and_overlap(
    pred_dir: Path,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Retrieve the tile size and overlap from saved configurations.

    Parameters
    ----------
    pred_dir : Path
        The directory containing the predictions from each method.
    tile_size : tuple[int, ...] | None
        If not None will directly return tile size.
    overlap : tuple[int, ...] | None
        If not None will directly return the overlap

    Returns
    -------
    tuple[int, ...]
        The tile size.
    tuple[int, ...]
        The overlap

    Raises
    ------
    FileNotFoundError
        If the inference_config.json file for the inner_tiling method cannot be found.
    """

    inference_config_path = pred_dir / "inner_tiling" / "inference_params.json"
    try:
        with open(inference_config_path) as f:
            inference_config = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find inference config at {inference_config_path}."
        ) from e

    tile_size = tuple(inference_config["tile_size"])
    overlap = tuple(inference_config["overlap"])

    return tile_size, overlap


def main():
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
