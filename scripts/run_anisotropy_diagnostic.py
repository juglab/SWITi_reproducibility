"""Per-axis anisotropy diagnostic on seam-free content of predicted volumes.

Experimental driver (argparse only) implementing
``agents_artifacts/anisotropy_diagnostic_spec.md``. For each predicted volume we

1. derive kept-region bounding boxes from ``(tile_size, overlap)`` via the
   existing :func:`tilartmetrics.gradient_test.tiles.enumerate_tiles`
   (Step 1);
2. keep only regions whose centre falls in the central ``central_fraction`` of
   the volume along every axis (Step 2, no occupancy/signal filtering);
3. extract one shared ``Wz x Wy x Wx`` window centred in each kept region and
   compute nearest-neighbour gradients along each axis from that *same* window
   via :func:`compute_gradients_3d` (Step 3) -- a ``W^3`` window yields
   ``W**2 * (W - 1)`` gradients per axis.

Gradients are pooled across all windows and volumes, separately per channel and
axis, and reported (Step 4) as: raw pooled arrays, quantiles + std, a histogram,
and a log-y overlay plot. Nothing else -- no cross-axis comparison, no tests, no
pooling decision (all downstream).

Axis order for every 3-tuple (``--tile_size``, ``--overlap``, ``--window``) is
``(z, y, x)`` matching the codebase ``(D, H, W)`` spatial convention.

Data layout mirrors ``run_gradient_test_on_dataset.py``: for a dataset ``D`` each
method stores a single ``predictions.npz`` whose keys are image names and whose
arrays squeeze to ``(C, D, H, W)``.

Example::

    python scripts/run_anisotropy_diagnostic.py \\
        --dataset CARE3D_liver \\
        --tile_size 5 64 64 --overlap 2 32 32 --window 3 3 3 \\
        --output_dir results/anisotropy/CARE3D_liver
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterator, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tilartmetrics.gradient_test.gradient_analysis import compute_gradients_3d
from tilartmetrics.gradient_test.tiles import Tile, enumerate_tiles


AXIS_LABELS = ("z", "y", "x")  # (D, H, W) spatial order
QUANTILE_PCTS = (1, 5, 25, 50, 75, 95, 99)
NUM_HIST_BINS = 100


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data location (mirrors run_gradient_test_on_dataset.py).
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. CARE3D_liver.")
    p.add_argument(
        "--results_root",
        type=Path,
        default=Path("/project/careamics/switi/results"),
        help="Root holding {dataset}/{predictions_subdir}/inner_tiling/predictions.npz.",
    )
    p.add_argument(
        "--predictions_subdir",
        default="predictions_MMSE64",
        help="Predictions folder under the dataset.",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap volumes for quick trials (default: all).",
    )
    # Geometry / parameters (each 3 ints, z y x order).
    p.add_argument(
        "--tile_size",
        required=True,
        type=int,
        nargs=3,
        metavar=("Tz", "Ty", "Tx"),
        help="Inner-tiling tile size per axis (z y x).",
    )
    p.add_argument(
        "--overlap",
        required=True,
        type=int,
        nargs=3,
        metavar=("Oz", "Oy", "Ox"),
        help="Inner-tiling overlap per axis (z y x).",
    )
    p.add_argument(
        "--window",
        required=True,
        type=int,
        nargs=3,
        metavar=("Wz", "Wy", "Wx"),
        help="Central sampling window per kept region (z y x); each W_a <= T_a - O_a.",
    )
    p.add_argument(
        "--central_fraction",
        type=float,
        default=0.5,
        help="Central image fraction (per axis) for region selection; 1.0 uses all.",
    )
    p.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="Channel indices to process (default: all channels, separately).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for the pooled arrays, summary JSON, and overlay plots.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data loading (mirrors run_gradient_test_on_dataset.py, generalised to 3D).
# --------------------------------------------------------------------------- #
def _ensure_cdhw(arr: np.ndarray) -> np.ndarray:
    """Squeeze to ``(C, D, H, W)``, promoting a bare ``(D, H, W)`` to one channel."""
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4:
        raise ValueError(f"expected (C, D, H, W) after squeeze, got shape {arr.shape}")
    return arr


def read_image_names(npz_path: Path, max_images: Optional[int]) -> list[str]:
    """Return the image names in a ``predictions.npz`` (reads the archive index)."""
    names = list(np.load(npz_path, allow_pickle=True).files)
    return names if max_images is None else names[:max_images]


def iter_prediction_volumes(
    npz_path: Path, image_names: list[str]
) -> Iterator[tuple[str, np.ndarray]]:
    """Lazily yield ``(name, (C, D, H, W))`` from a ``predictions.npz``."""
    with np.load(npz_path, allow_pickle=True) as data:
        for name in image_names:
            yield name, _ensure_cdhw(data[name])


# --------------------------------------------------------------------------- #
# Geometry / sampling.
# --------------------------------------------------------------------------- #
def validate_geometry(
    tile_size: Sequence[int], overlap: Sequence[int], window: Sequence[int]
) -> list[int]:
    """Validate ``W_a <= S_a`` per axis and warn on odd overlaps; return strides."""
    strides: list[int] = []
    for a, (t, o, w) in enumerate(zip(tile_size, overlap, window)):
        s = t - o
        if o % 2 != 0:
            warnings.warn(
                f"axis {AXIS_LABELS[a]!r}: overlap={o} is odd; the halo O/2 is "
                "asymmetric (floor/ceil) -- kept-region derivation may be off by "
                "one at seams.",
                stacklevel=2,
            )
        if w > s:
            raise ValueError(
                f"axis {AXIS_LABELS[a]!r}: window W={w} exceeds kept-region size "
                f"S=T-O={t}-{o}={s}; the central window must stay inside a single "
                f"kept region (e.g. for CBG-Z18 T_z=5, O_z=2 caps W_z<=3)."
            )
        strides.append(s)
    return strides


def select_central_tiles(
    tiles: Sequence[Tile], shape: Sequence[int], central_fraction: float
) -> list[Tile]:
    """Keep tiles whose region centre lies in the central fraction of every axis."""
    if not (0.0 < central_fraction <= 1.0):
        raise ValueError(f"central_fraction must be in (0, 1]; got {central_fraction}")
    lo_f = 0.5 - central_fraction / 2.0
    hi_f = 0.5 + central_fraction / 2.0
    bounds = [(lo_f * size, hi_f * size) for size in shape]

    kept: list[Tile] = []
    for tile in tiles:
        centres = [(lo + hi) / 2.0 for (lo, hi) in tile.ranges]
        if all(b_lo <= c <= b_hi for c, (b_lo, b_hi) in zip(centres, bounds)):
            kept.append(tile)
    return kept


def extract_window(
    volume: np.ndarray, ranges: Sequence[tuple[int, int]], window: Sequence[int]
) -> np.ndarray:
    """Extract the ``window`` cube centred in ``ranges`` (clamped inside the region).

    ``volume`` is a single-channel ``(D, H, W)`` array; ``ranges`` are the kept
    region's per-axis ``(lo, hi)`` bounds. The window is centred on the region
    and clamped so it stays fully inside the seam-free region.
    """
    slices: list[slice] = []
    for (lo, hi), w in zip(ranges, window):
        centre = (lo + hi) // 2
        start = centre - w // 2
        start = min(max(start, lo), hi - w)  # keep [start, start+w) inside [lo, hi)
        slices.append(slice(start, start + w))
    return volume[tuple(slices)]


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def summarise_axis(values: np.ndarray, bin_edges: np.ndarray) -> dict:
    """Compute n, std, quantiles, and histogram counts for one pooled axis array."""
    counts, _ = np.histogram(values, bins=bin_edges)
    quantiles = np.percentile(values, QUANTILE_PCTS)
    return {
        "n": int(values.size),
        "std": float(np.std(values)),
        "quantiles": {f"p{p}": float(q) for p, q in zip(QUANTILE_PCTS, quantiles)},
        "hist_counts": counts.astype(int).tolist(),
    }


def plot_overlay(
    channel: int,
    pooled: dict[str, np.ndarray],
    bin_edges: np.ndarray,
    out_path: Path,
    *,
    xlabel: str = "nearest-neighbour gradient",
    title: str = "Per-axis gradient distribution",
) -> None:
    """Overlay the three axes' histograms on shared bins with a log y-axis."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in AXIS_LABELS:
        ax.hist(
            pooled[label],
            bins=bin_edges,
            histtype="step",
            linewidth=1.5,
            label=f"grad_{label} (n={pooled[label].size})",
        )
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (log)")
    ax.set_title(f"{title} -- channel {channel}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def zscore_per_axis(pooled: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Standardise each axis independently: ``(g - mean) / std`` per axis.

    A diagnostic view of what per-axis standardisation (one candidate way to make
    the axes poolable) would do -- if the axes differ only in scale they collapse
    onto each other; a residual shape difference (e.g. heavier z tails in std
    units) would survive.
    """
    out: dict[str, np.ndarray] = {}
    for label, g in pooled.items():
        g = g.astype(np.float64)
        std = g.std()
        out[label] = (g - g.mean()) / std if std > 0 else g - g.mean()
    return out


def robust_scale_per_axis(pooled: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Standardise each axis by median and MAD: ``(g - median) / (1.4826 * MAD)``.

    A tail-robust counterpart to :func:`zscore_per_axis`. The mean/std z-score is
    dominated by the heavy tails, so a scale difference driven by outliers can
    still look like a good collapse; median/MAD centres and scales on the bulk,
    giving a stricter check of whether the axes share a common shape. The
    ``1.4826`` factor makes MAD a consistent estimator of the std for Gaussian
    data, so the x-axis reads in comparable "robust sigma" units.
    """
    out: dict[str, np.ndarray] = {}
    for label, g in pooled.items():
        g = g.astype(np.float64)
        median = np.median(g)
        mad = np.median(np.abs(g - median))
        scale = 1.4826 * mad
        out[label] = (g - median) / scale if scale > 0 else g - median
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_geometry(args.tile_size, args.overlap, args.window)

    npz_path = (
        args.results_root
        / args.dataset
        / args.predictions_subdir
        / "inner_tiling"
        / "predictions.npz"
    )
    image_names = read_image_names(npz_path, args.max_images)
    print(f"=== {args.dataset} / inner_tiling: {len(image_names)} volumes ===")

    # Pooled gradients: channel -> axis label -> list of 1-D arrays.
    pools: dict[int, dict[str, list[np.ndarray]]] = {}
    total_windows = 0

    for name, volume in iter_prediction_volumes(npz_path, image_names):
        n_channels = volume.shape[0]
        channels = args.channels if args.channels is not None else list(range(n_channels))
        spatial_shape = volume.shape[1:]  # (D, H, W)

        tiles = enumerate_tiles(spatial_shape, args.tile_size, args.overlap)
        kept = select_central_tiles(tiles, spatial_shape, args.central_fraction)
        total_windows += len(kept) * len(channels)
        print(
            f"  {name}: shape={volume.shape}, {len(tiles)} kept-regions, "
            f"{len(kept)} selected (central_fraction={args.central_fraction})"
        )

        for c in channels:
            if not (0 <= c < n_channels):
                raise ValueError(f"{name}: channel {c} out of range for C={n_channels}")
            axis_pool = pools.setdefault(c, {label: [] for label in AXIS_LABELS})
            vol_c = volume[c]
            for tile in kept:
                window = extract_window(vol_c, tile.ranges, args.window)
                grads = compute_gradients_3d(window)  # (g_z, g_y, g_x)
                for label, g in zip(AXIS_LABELS, grads):
                    axis_pool[label].append(g.ravel())

    if not pools:
        raise RuntimeError("No windows sampled; check geometry / central_fraction.")

    # Pool, summarise, and write outputs per channel.
    summary: dict = {
        "dataset": args.dataset,
        "method": "inner_tiling",
        "tile_size": list(args.tile_size),
        "overlap": list(args.overlap),
        "window": list(args.window),
        "central_fraction": args.central_fraction,
        "n_volumes": len(image_names),
        "quantile_pcts": list(QUANTILE_PCTS),
        "channels": {},
    }

    for c in sorted(pools):
        pooled = {
            label: np.concatenate(pools[c][label]) for label in AXIS_LABELS
        }
        # Shared bin edges across the three axes of this channel.
        all_vals = np.concatenate([pooled[label] for label in AXIS_LABELS])
        bin_edges = np.linspace(all_vals.min(), all_vals.max(), NUM_HIST_BINS + 1)

        np.savez_compressed(
            out_dir / f"anisotropy_grads_c{c}.npz",
            grad_z=pooled["z"],
            grad_y=pooled["y"],
            grad_x=pooled["x"],
        )
        plot_overlay(c, pooled, bin_edges, out_dir / f"anisotropy_hist_c{c}.png")

        # Diagnostic overlay of per-axis z-scored gradients: shows whether the
        # axes collapse under per-axis standardisation (scale-only difference).
        zscored = zscore_per_axis(pooled)
        z_vals = np.concatenate([zscored[label] for label in AXIS_LABELS])
        z_edges = np.linspace(z_vals.min(), z_vals.max(), NUM_HIST_BINS + 1)
        plot_overlay(
            c,
            zscored,
            z_edges,
            out_dir / f"anisotropy_hist_zscored_c{c}.png",
            xlabel="per-axis z-scored gradient  (g - mean) / std",
            title="Per-axis z-scored gradient distribution",
        )

        # Tail-robust counterpart: centre/scale each axis by median and MAD.
        robust = robust_scale_per_axis(pooled)
        r_vals = np.concatenate([robust[label] for label in AXIS_LABELS])
        r_edges = np.linspace(r_vals.min(), r_vals.max(), NUM_HIST_BINS + 1)
        plot_overlay(
            c,
            robust,
            r_edges,
            out_dir / f"anisotropy_hist_robust_c{c}.png",
            xlabel="per-axis robust-scaled gradient  (g - median) / (1.4826*MAD)",
            title="Per-axis median/MAD-scaled gradient distribution",
        )

        ch_summary = {"hist_bin_edges": bin_edges.tolist(), "axes": {}}
        for label in AXIS_LABELS:
            ch_summary["axes"][label] = summarise_axis(pooled[label], bin_edges)
        summary["channels"][str(c)] = ch_summary

        print(f"\n  channel {c} (per-axis std / median / p1 / p99):")
        for label in AXIS_LABELS:
            s = ch_summary["axes"][label]
            print(
                f"    {label}: n={s['n']:>8} std={s['std']:.4g} "
                f"p50={s['quantiles']['p50']:.4g} "
                f"p1={s['quantiles']['p1']:.4g} p99={s['quantiles']['p99']:.4g}"
            )

    with open(out_dir / "anisotropy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\nSampled {total_windows} windows across {len(image_names)} volumes. "
        f"Wrote arrays + summary.json + overlay plots to {out_dir}"
    )


if __name__ == "__main__":
    main()
