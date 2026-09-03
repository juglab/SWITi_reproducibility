"""Build CAREamics configuration objects for MicroSplit inference scripts.

The helpers in this module read training configuration data from `config.yaml`
files and return prediction, model, and algorithm configuration objects used by
the inference scripts. The `config.yaml` is the old original configuration file.
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
    """Load a MicroSplit training configuration file.

    Parameters
    ----------
    path : str or Path
        Path to a YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration data.
    """
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
    """Build a prediction data configuration from training config data.

    Parameters
    ----------
    config_data : dict
        Data section of a MicroSplit training configuration. It must contain
        `image_size`, `multiscale_lowres_count`, and `padding_mode`, and may
        contain `mode_3D` and `depth3D` for 3D experiments.
    overlap : list of int
        Overlap per spatial dimension (length 2 for 2D, length 3 for 3D).
    stride : list of int or None, default=None
        Sliding-window stride. If `None`, classical inner tiling is configured.
    input_means, input_stds : list of float
        Per-input-channel normalization stats.
    target_means, target_stds : list of float
        Per-target-channel normalization stats (used to denormalize predictions).
    batch_size : int, default=1
        Prediction batch size.

    Returns
    -------
    MicroSplitDataConfig
        Prediction data configuration.
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
    """Build an LVAE model configuration from training config data.

    Parameters
    ----------
    config_data : dict
        Parsed MicroSplit training configuration.

    Returns
    -------
    LVAEConfig
        Model configuration for the MicroSplit algorithm.
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
    """Build a MicroSplit algorithm configuration.

    Parameters
    ----------
    config_data : dict
        Parsed MicroSplit training configuration.

    Returns
    -------
    MicroSplitAlgorithm
        Algorithm configuration containing the model configuration.
    """
    return MicroSplitAlgorithm(model=get_model_config(config_data))


def _resolve_output_channels(config_data: dict) -> int:
    """Resolve the number of output target channels.

    Parameters
    ----------
    config_data : dict
        Parsed MicroSplit training configuration.

    Returns
    -------
    int
        Number of target channels.

    Raises
    ------
    KeyError
        If no supported output-channel field is present.
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
        "Could not resolve output channels from config data: none of "
        "`model.num_targets`, `data.target_idx_list`, `data.num_channels` "
        "is present."
    )
