"""Pydantic-config factories for MicroSplit inference scripts.

Holds the pkl-driven factories that build the configs feeding a
`MicroSplitModule`:

- :func:`pkl_load` — load a legacy training-config dump.
- :func:`get_predict_config` — `MicroSplitDataConfig` (data side; picks tiled vs
  sliding-window patching based on whether `stride` is given).
- :func:`get_model_config` — `LVAEConfig` (architecture).
- :func:`get_loss_config` — `LVAELossConfig`. Loss is not actually used at
  inference time, but :class:`VAEBasedAlgorithm` requires it to validate. We
  hardcode the loss type to ``"denoisplit_musplit"`` (the value used by every
  experiment we predict on); kl-type is read from the pkl for completeness.
- :func:`get_likelihood_config` — `GaussianLikelihoodConfig`. Like the loss, not
  consumed at predict time, but supplied so :class:`VAEBasedAlgorithm` matches
  what the checkpoint was trained with.
- :func:`create_algorithm_config` — `VAEBasedAlgorithm` assembled from the
  three above.
"""
from pathlib import Path
from typing import Literal

import yaml

from careamics.config.algorithms import MicroSplitAlgorithm
from careamics.config.architectures import LVAEConfig
from careamics.config.data.data_config import (
    Mode,
)
from careamics.config.data.patching_strategies import SwitiPatchingConfig, TiledPatchingConfig
from careamics.config.data.microsplit_data_config import MicroSplitDataConfig
from careamics.config.data.normalization_config import MeanStdConfig

# Legacy `nonlin` strings are lowercase; `LVAEConfig.nonlinearity` is a
# capitalised Literal.
NonlinLiteral = Literal[
    "None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU", "ELU"
]
_NONLIN_MAP: dict[str, NonlinLiteral] = {
    "elu": "ELU",
    "relu": "ReLU",
    "leakyrelu": "LeakyReLU",
    "leaky_relu": "LeakyReLU",
    "sigmoid": "Sigmoid",
    "softmax": "Softmax",
    "tanh": "Tanh",
    "none": "None",
}


def load_config_data(path: str | Path) -> dict:
    """Load a legacy MicroSplit training config dump."""
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return config_data


def get_predict_config(
    config_data: dict,
    *,
    overlap: list[int],
    stride: list[int] | None = None,
    input_means: list[float],
    input_stds: list[float],
    target_means: list[float],
    target_stds: list[float],
    batch_size: int = 1,
) -> MicroSplitDataConfig:
    """Build a prediction `MicroSplitDataConfig` from a legacy training-config dump.

    Parameters
    ----------
    pkl_data : dict
        Legacy MicroSplit training config (loaded with :func:`pkl_load`).
        Must carry `image_size`, `multiscale_lowres_count`, `padding_mode`, and
        optionally `mode_3D` / `depth3D` for 3D experiments.
    overlap : list of int
        Overlap per spatial dimension (length 2 for 2D, length 3 for 3D).
    stride : list of int or None, default=None
        If `None`, a classical `TiledPatchingConfig` is used (inner tiling, no
        overlap on the kept region). If provided, a
        `SlidingWindowTiledPatchingConfig` is used (dense overlap averaging — see
        :class:`careamics.dataset.patching.SlidingWindowTiledPatching`).
    input_means, input_stds : list of float
        Per-input-channel normalization stats.
    target_means, target_stds : list of float
        Per-target-channel normalization stats (used to denormalize predictions).
    batch_size : int, default=1
        Prediction batch size.

    Returns
    -------
    MicroSplitDataConfig
        Configuration ready to be passed to
        :func:`careamics.dataset.factory.create_microsplit_pred_dataset`.
    """
    is_3d = config_data.get("mode_3D", False)
    axes = "CZYX" if is_3d else "CYX"
    img = config_data["image_size"]
    patch_size = [config_data["depth3D"], img, img] if is_3d else [img, img]

    patching = (
        SwitiPatchingConfig(patch_size=patch_size, overlaps=overlap, stride=stride)
        if stride is not None
        else TiledPatchingConfig(patch_size=patch_size, overlaps=overlap)
    )

    return MicroSplitDataConfig(
        mode=Mode.PREDICTING,
        data_type="tiff",
        axes=axes,
        patching=patching,
        normalization=MeanStdConfig(
            input_means=input_means,
            input_stds=input_stds,
            target_means=target_means,
            target_stds=target_stds,
        ),
        multiscale_count=config_data["multiscale_lowres_count"],
        padding_mode=config_data["padding_mode"],
        batch_size=batch_size,
        augmentations=[],  # predict mode: no augs
    )


def get_model_config(config_data: dict) -> LVAEConfig:
    """Build an `LVAEConfig` from a legacy MicroSplit training-config dump.

    Architecture fields come from `pkl_data["model"]`; spatial / multiscale
    fields come from `pkl_data["data"]`. Output channels are resolved by trying,
    in order: `model.num_targets`, `len(data.target_idx_list)`,
    `data.num_channels` — covers both paired and multiplexed experiments.
    """
    data = config_data["data"]
    model = config_data["model"]

    is_3d = bool(data.get("mode_3D", False))
    img = int(data["image_size"])
    if is_3d:
        input_shape = (int(data["depth3D"]), img, img)
        strides = [1, 2, 2]
    else:
        input_shape = (img, img)
        strides = [2, 2]

    nonlin_raw = str(model.get("nonlin", "ELU"))
    nonlinearity = _NONLIN_MAP.get(nonlin_raw.lower(), "ELU")

    return LVAEConfig(
        architecture="LVAE",
        input_shape=input_shape,
        encoder_conv_strides=strides,
        decoder_conv_strides=strides,
        multiscale_count=int(data["multiscale_lowres_count"]),
        z_dims=list(model.get("z_dims", [128, 128, 128, 128])),
        output_channels=_resolve_output_channels(config_data),
        nonlinearity=nonlinearity,
        predict_logvar=True if model.get("predict_logvar", False) else False,
    )


def create_algorithm_config(config_data: dict) -> MicroSplitAlgorithm:
    """Assemble a `VAEBasedAlgorithm` from a legacy training-config dump.

    Composes :func:`get_model_config`, :func:`get_loss_config` and
    :func:`get_likelihood_config`. Algorithm is always ``"microsplit"``
    (CAREamics's umbrella label for muSplit / denoiSplit / denoiSplit-muSplit
    training).
    """
    return MicroSplitAlgorithm(model=get_model_config(config_data))


def _resolve_output_channels(config_data: dict) -> int:
    """Resolve the number of output (target) channels from a legacy pkl dump.

    Tries, in order: `model.num_targets`, `len(data.target_idx_list)`,
    `data.num_channels`. Covers both paired (HT_LIF24) and multiplexed
    (CARE3D / PaviaATN) experiments.
    """
    model = config_data.get("model", {})
    if model.get("num_targets") is not None:
        return int(model["num_targets"])
    data = config_data.get("data", {})
    target_idx_list = data.get("target_idx_list")
    if target_idx_list is not None:
        return len(target_idx_list)
    if data.get("num_channels") is not None:
        return int(data["num_channels"])
    raise KeyError(
        "Could not resolve output channels from pkl: none of "
        "`model.num_targets`, `data.target_idx_list`, `data.num_channels` "
        "is present."
    )
