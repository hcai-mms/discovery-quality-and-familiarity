import numpy as np
import copy
import json
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


def to_python_scalar(value):
    """
    Convert tensors and numpy scalars to Python scalars when possible.
    """

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()

        if value.numel() == 1:
            return value.item()

        return None

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (int, float)):
        return value

    return None


def flatten_dict(d, parent_key="", sep="/"):
    """
    Flatten nested dictionaries for metric logging.
    """

    items = {}

    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value

    return items


def extract_scalar_metrics(metrics):
    """
    Keep only scalar metrics suitable for TensorBoard and W&B.
    """

    flat_metrics = flatten_dict(metrics)
    scalar_metrics = {}

    for key, value in flat_metrics.items():
        scalar_value = to_python_scalar(value)

        if scalar_value is not None:
            scalar_metrics[key] = scalar_value

    return scalar_metrics


def log_to_tensorboard(writer, metrics, epoch, prefix):
    """
    Log scalar metrics to TensorBoard.
    """

    if writer is None:
        return

    scalar_metrics = extract_scalar_metrics(metrics)

    for key, value in scalar_metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, epoch)


# def log_to_wandb(metrics, epoch, prefix):
#     """
#     Log scalar metrics to Weights & Biases.
#     """

#     if wandb.run is None:
#         return

#     scalar_metrics = extract_scalar_metrics(metrics)

#     wandb.log(
#         {
#             f"{prefix}/{key}": value
#             for key, value in scalar_metrics.items()
#         },
#         step=epoch,
#     )