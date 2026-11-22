"""Load NWPU-MOC training checkpoints and standalone model weights."""
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
from numpy._core import multiarray
import torch


def read_checkpoint(path):
    # NWPU-MOC checkpoints may include NumPy metrics in their training record.
    numpy_globals = [
        np.ndarray, np.dtype, multiarray.scalar, multiarray._reconstruct,
        (multiarray.scalar, "numpy.core.multiarray.scalar"),
        (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        *(type(np.dtype(dtype)) for dtype in (np.float32, np.float64, np.int32, np.int64)),
    ]
    with torch.serialization.safe_globals(numpy_globals):
        return torch.load(path, weights_only=True, map_location="cpu")


def load_counter_weights(path, counter):
    checkpoint = read_checkpoint(path)
    state = checkpoint.get("net", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state, Mapping) or not state or not all(isinstance(key, str) for key in state):
        raise ValueError(f"Expected a model state_dict or a checkpoint containing 'net': {path}")

    def strip_parallel(key):
        return ".".join(part for part in key.split(".") if part != "module")

    normalized = OrderedDict((strip_parallel(key), value) for key, value in state.items())
    if len(normalized) != len(state):
        raise ValueError(f"Duplicate model keys after removing DataParallel prefixes: {path}")
    metadata = {strip_parallel(key): value for key, value in getattr(state, "_metadata", {}).items()}

    normalized._metadata = metadata
    if any(key.startswith("CCN.") for key in normalized):
        if isinstance(counter.CCN, torch.nn.DataParallel):
            counter.CCN.module.load_state_dict(
                OrderedDict((key.removeprefix("CCN."), value) for key, value in normalized.items() if key.startswith("CCN.")),
                strict=True,
            )
        else:
            counter.load_state_dict(normalized, strict=True)
    else:
        network = counter.CCN.module if isinstance(counter.CCN, torch.nn.DataParallel) else counter.CCN
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(normalized, "network.")
        network.load_state_dict(normalized, strict=True)
