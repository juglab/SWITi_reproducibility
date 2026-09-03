"""MicroSplit Lightning-module factory for inference scripts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from careamics.lightning.modules.microsplit_module import MicroSplitModule

from .config_factory import create_algorithm_config, load_config_data

if TYPE_CHECKING:
    pass


# Top-level keys that don't map onto the NG MicroSplitModule and are dropped
# before loading. `noiseModel.*` and `likelihood_NM.*` carry the noise-model +
# NM-likelihood weights.
# NOTE: Noise models are only used in the loss during training, so they are not needed
# for SWITi inference.
# A refactoring of the LVAE in CAREamics has removed the likelihood from the model.
_DROP_KEY_PREFIXES: tuple[str, ...] = ("noiseModel.", "likelihood_NM.")

# Suffixes dropped from each key. `num_batches_tracked` is a BatchNorm
# bookkeeping buffer that PyTorch saves by default but the NG LVAE's BN layers
# don't register, so it shows up as "unexpected" at load time.
_DROP_KEY_SUFFIXES: tuple[str, ...] = (".num_batches_tracked",)


def convert_legacy_state_dict(state_dict: dict) -> dict:
    """Return checkpoint weights with keys accepted by `MicroSplitModule`.

    Parameters
    ----------
    state_dict : dict
        Raw checkpoint state dictionary.

    Returns
    -------
    dict
        State dictionary compatible with the loaded MicroSplit module.
    """
    converted: dict = {}
    for key, value in state_dict.items():
        if any(key.startswith(p) for p in _DROP_KEY_PREFIXES):
            continue
        if any(key.endswith(s) for s in _DROP_KEY_SUFFIXES):
            continue
        converted[f"model.{key}"] = value
    return converted


def build_microsplit_module(
    ckpt_path: str | Path,
    config_path: str | Path,
    device: "torch.device | str | None" = None,
) -> MicroSplitModule:
    """Instantiate a `MicroSplitModule` and load weights from a checkpoint.

    Parameters
    ----------
    ckpt_path : str or Path
        Path to the Lightning checkpoint file.
    config_path : str or Path
        Path to the MicroSplit training configuration YAML file.
    device : torch.device or str or None, default=None
        If provided, the module is moved to this device.

    Returns
    -------
    MicroSplitModule
        Module in eval mode with weights loaded.
    """
    config_data = load_config_data(config_path)
    algorithm_config = create_algorithm_config(config_data)

    module = MicroSplitModule(algorithm_config=algorithm_config)

    ckpt = torch.load(
        Path(ckpt_path), map_location="cpu", weights_only=False
    )
    state_dict = convert_legacy_state_dict(ckpt["state_dict"])
    module.load_state_dict(state_dict, strict=True)

    module.eval()
    if device is not None:
        module = module.to(device)
    return module
