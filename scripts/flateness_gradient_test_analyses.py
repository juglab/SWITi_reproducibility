"""Flatness-stratified rejection analysis.

Adjudicates *why* SWITi rejects in flat regions: does the per-tile test fire
in flat tiles because of a real (texture-unmasked) SWITi signal, or because the
narrow gradient distribution inflates the false-positive rate there?

The discriminator is to stratify the per-tile rejection rate by a *shared*
per-tile texture (computed once from the ground truth, so every method sits on
the same x-axis) and overlay the three curves:

- GT -> known null (no seams); should sit near alpha at *every* texture.
- SWITi —> the method under question.
- Inner Tiling —> the strong-artifact reference.

If GT stays flat at alpha while SWITi climbs in the low-texture bins, the
flat-region positives are a real SWITi signal that flatness merely unmasks. 
If GT *also* climbs at low texture, the narrow-distribution regime genuinely
inflates the false-positive rate.

Reports are cached to ``--cache-dir`` so the plot can be re-styled without
re-running the (~100 s/method) tile scan.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import tifffile as tiff

from tilartmetrics.gradient_test.analysis import run_gradient_analysis
from tilartmetrics.gradient_test.plotting import (
    plot_flatness_stratified_rejection,
)

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

ROOT = Path("/project/careamics/switi")
PRED_DIR = ROOT / "results/PaviaATN/predictions"
PATHS = {
    "Inner Tiling": PRED_DIR / "inner_tiling/predictions.npz",
    "SWITi": PRED_DIR / "sw_inner_tiling/predictions.npz",
}
GT_PATH = ROOT / "data/PaviaATN/targets/test/test_img01.tif"

TILE_SIZE = [64, 64]
OVERLAP = [32, 32]
STATISTIC = "js"
ALPHA = 0.05

OUT_DIR = Path(__file__).resolve().parents[1] / "agents_artifacts" / "figures"


def _load_image(method: str, image_key: str) -> np.ndarray:
    """Load one ``(C, H, W)`` prediction image for a method."""
    data = np.load(PATHS[method], allow_pickle=True)
    return data[image_key].squeeze()


def _image_report(
    image_chw: np.ndarray,
    channel: int,
    method_name: str,
    crop,
    n_permutations: int,
    cache_path: Path | None,
):
    """Run (or load cached) the per-tile test, returning an ImageReport."""
    if cache_path is not None and cache_path.exists():
        print(f"  [{method_name}] loading cached report: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    img = image_chw
    if crop is not None:
        (y0, y1), (x0, x1) = crop
        img = img[:, y0:y1, x0:x1]

    report = run_gradient_analysis(
        img[np.newaxis, ...],
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        channel=channel,
        method_name=method_name,
        statistic=STATISTIC,
        n_permutations=n_permutations,
        alpha=ALPHA,
        save_dir=None,
    )
    ir = report.images[0]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(ir, f)
        print(f"  [{method_name}] cached report -> {cache_path}")
    return ir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--image-key", default="test_img01")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument(
        "--texture",
        choices=["grad", "intensity"],
        default="grad",
        help="define tile texture from gradient magnitude (default) or intensity",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("Y0", "Y1", "X0", "X1"),
        default=None,
        help="optional crop (quick smoke test); omit for the full image",
    )
    parser.add_argument("--cache-dir", default=str(OUT_DIR / "_reports_cache"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    crop = None
    if args.crop is not None:
        crop = ((args.crop[0], args.crop[1]), (args.crop[2], args.crop[3]))
    cache_dir = Path(args.cache_dir)
    tag = f"ch{args.channel}_{args.image_key}_R{args.n_permutations}" + (
        f"_crop{'-'.join(map(str, args.crop))}" if crop is not None else "_full"
    )

    # --- ground truth (shared texture reference + known null) ---------------
    gt = tiff.imread(GT_PATH)  # (C, H, W)
    gt_img = gt[:, crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]] if crop else gt

    reports = {}
    print("== Ground truth (null reference) ==")
    reports["GT"] = _image_report(
        gt, args.channel, "GT", crop, args.n_permutations,
        cache_dir / f"GT_{tag}.pkl",
    )
    for method in PATHS:
        print(f"== {method} ==")
        img_chw = _load_image(method, args.image_key)
        reports[method] = _image_report(
            img_chw, args.channel, method, crop, args.n_permutations,
            cache_dir / f"{method.replace(' ', '_')}_{tag}.pkl",
        )

    # report the plain (unstratified) rejection rates for the record
    print("\nOverall frac_rejected:")
    for name, ir in reports.items():
        print(f"  {name:14s} {ir.frac_rejected:.3f}")

    out_path = Path(args.out_dir) / f"flatness_stratified_rejection_{tag}.png"
    plot_flatness_stratified_rejection(
        reports,
        gt_img[args.channel],
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        alpha=ALPHA,
        n_bins=args.n_bins,
        use_gradient_magnitude=(args.texture == "grad"),
        title=(
            f"Rejection vs. tile texture — {args.image_key} ch{args.channel} "
            f"({STATISTIC.upper()})"
        ),
        save_path=out_path,
    )
    print(f"\nDone. Figure: {out_path}")


if __name__ == "__main__":
    main()
