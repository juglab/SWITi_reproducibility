"""MicroSplit — classical tiled inference (CL-args entry point).

Headless companion to :mod:`scripts.microsplit_tiled_inference`. Loads a
pre-trained MicroSplit checkpoint, runs prediction with classical inner tiling
(`TiledPatching`, non-overlapping kept regions), stitches via the canonical
`convert_prediction(..., tiled=True)` path, and saves a single `.npz` keyed by
input-image identifier.

Run from the repo root:

    python -m scripts.microsplit_tiled_predict --dataset HT_LIF24_5ms
    python -m scripts.microsplit_tiled_predict --dataset CARE3D_liver \\
        --overlap 0 32 32 --mmse-count 50 --batch-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from lightning import Trainer

from careamics.dataset.factory.microsplit_factory import create_microsplit_pred_dataset
from careamics.lightning.data.data_module_utils import initialize_data_pair
from careamics.lightning.prediction.convert_prediction import convert_prediction

from utils.config_factory import load_config_data, get_predict_config
from utils.io_utils import npz_key, save_inference_params, save_predictions_npz
from utils.microsplit_factory import build_microsplit_module
from utils.stats import load_or_compute_stats


def main(args: argparse.Namespace) -> Path:
    """Run end-to-end tiled inference for a single experiment.

    Returns the path of the written NPZ file.
    """
    data_dir = args.data_root / args.dataset
    ckpt_path = args.ckpt_root / args.dataset / "BaselineVAECL_best.ckpt"
    config_path = args.ckpt_root / args.dataset / "config.yaml"
    save_dir = (
        args.out_root
        / args.dataset
        / f"predictions_MMSE{args.mmse_count}"
        / "inner_tiling"
    )

    config_data = load_config_data(config_path)["data"]
    is_3d = bool(config_data.get("mode_3D", False))
    stats = load_or_compute_stats(
        name=args.dataset,
        data_dir=data_dir,
        is_3d=is_3d,
        force=args.force_recompute_stats,
    )
    pred_config = get_predict_config(
        config_data,
        overlap=args.overlap,
        stride=None,
        batch_size=args.batch_size,
        **stats,
    )
    pred_data_validated, _ = initialize_data_pair(
        data_type=pred_config.data_type, input_data=data_dir / "inputs" / args.split
    )
    dataset = create_microsplit_pred_dataset(
        config=pred_config, input_data=pred_data_validated
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=default_collate,
        **pred_config.pred_dataloader_params,
    )

    model = build_microsplit_module(ckpt_path=ckpt_path, config_path=config_path)
    model.n_samples = args.mmse_count
    trainer = Trainer()
    predictions = trainer.predict(model=model, dataloaders=dataloader)
    prediction_means = [pred[0] for pred in predictions] # ignore standard deviation

    preds_list, sources = convert_prediction(
        prediction_means, tiled=True, restore_shape=False
    )
    print(f"stitched {len(preds_list)} image(s)")

    results: dict[str, NDArray] = {}
    for data_idx, pred in enumerate(preds_list):
        source = sources[data_idx] if sources else "array"
        results[npz_key(source, data_idx)] = pred

    out_path = save_predictions_npz(results, save_dir)
    print(f"wrote {len(results)} prediction(s) to {out_path}")

    params_path = save_inference_params(
        {
            "tile_size": dataset.config.patching.patch_size,
            "overlap": args.overlap,
            "mmse_count": args.mmse_count,
            "ckpt_path": ckpt_path,
            "data_dir": data_dir,
        },
        save_dir,
    )
    print(f"wrote inference params to {params_path}")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="microsplit-tiled-predict",
        description=(
            "Run MicroSplit classical tiled inference on one experiment, "
            "saving the per-image stitched predictions as a single .npz."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="dataset name; resolves <data_root>/<dataset>/ and <ckpt_root>/<dataset>/",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="which on-disk split to predict on",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/project/careamics/switi/data"),
        help="root of <dataset>/{inputs,targets}/{train,val,test}/*.tif",
    )
    p.add_argument(
        "--ckpt-root",
        type=Path,
        default=Path("/project/careamics/switi/ckpts"),
        help="root of <dataset>/{BaselineVAECL_best.ckpt, config.pkl}",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("/project/careamics/switi/results"),
        help="root for predictions; output written to "
        "<out_root>/<dataset>/predictions/inner_tiling/predictions.npz",
    )
    p.add_argument(
        "--overlap",
        type=int,
        nargs="+",
        default=[32, 32],
        metavar="N",
        help="tile overlap per spatial axis (length 2 for 2D, 3 for 3D)",
    )
    p.add_argument(
        "--mmse-count",
        type=int,
        default=64,
        help="number of stochastic forward passes per tile",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="prediction batch size",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help='"auto" picks cuda when available, falls back to cpu',
    )
    p.add_argument(
        "--force-recompute-stats",
        action="store_true",
        help="bypass the <data_dir>/stats.json cache",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
