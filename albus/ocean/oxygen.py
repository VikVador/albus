r"""Metrics for evaluating oxygen-related indicators."""

__all__ = [
    "hypoxia",
]

import torch

from math import prod
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor
from tqdm import tqdm

from albus.tools import apply_mask, axes


# fmt: off
def hypoxia(
    x: Tensor,
    y: Tensor,
    dims: str,
    reduce: str,
    thresholds: list[float],
    thresholds_curve: list[float] | None = None,
    mask: Tensor | None = None,
) -> dict[str, Tensor]:
    r"""Compute classification metrics for a hypoxia (oxygen < threshold) detection problem.

    Arguments:
        x                : Ground truth tensor, broadcastable with `y`.
        y                : Predicted tensor.
        dims             : Space-separated axis names describing the shape of `y`, e.g. "T N C Y X".
        reduce           : Space-separated subset of `dims` to flatten and score jointly, e.g. "N Y X".
        thresholds       : [x_threshold, y_threshold], applied to `x` and `y` respectively.
        thresholds_curve : Thresholds swept over `y` to build the ROC/precision-recall curves.
        mask             : Boolean-like tensor, broadcastable with `x`, zero where entries are invalid.

    Returns:
        A dict of tensors, each with the axes in `reduce` collapsed:
        - `accuracy`, `balanced_accuracy`, `f1`, `precision`, `recall`.
        - if `thresholds_curve` is given: `roc_fpr`, `roc_tpr`, `roc_auc`,
          `pr_precision`, `pr_recall`, `pr_auc` (the curves stacked along a
          leading axis of size `len(thresholds_curve)`).

    Example:
        >>> x = torch.rand(7, 1, 2, 128, 256) * 400  # (T, N, C, Y, X)
        >>> y = torch.rand(7, 4, 2, 128, 256) * 400  # (T, N, C, Y, X)
        >>> out = hypoxia(
        ...     x=x,
        ...     y=y,
        ...     dims="T N C Y X",
        ...     reduce="N Y X",
        ...     thresholds=[63.0, 63.0],
        ... )
        >>> out["accuracy"].shape
        torch.Size([7, 2])
    """
    x_threshold, y_threshold = thresholds
    reduce_axes              = axes(dims, reduce)
    tail                     = tuple(range(-len(reduce_axes), 0))

    y          = apply_mask(y, mask)
    x          = apply_mask(x, mask).expand_as(y).movedim(reduce_axes, tail)
    y          = y.movedim(reduce_axes, tail)
    kept_shape = x.shape[: -len(reduce_axes)]
    n_kept     = prod(kept_shape)
    x          = x.reshape(n_kept, -1)
    y          = y.reshape(n_kept, -1)

    accuracy          = torch.full((n_kept,), torch.nan)
    balanced_accuracy = torch.full((n_kept,), torch.nan)
    f1                = torch.full((n_kept,), torch.nan)
    precision         = torch.full((n_kept,), torch.nan)
    recall            = torch.full((n_kept,), torch.nan)

    if thresholds_curve is not None:
        n_curve      = len(thresholds_curve)
        roc_fpr      = torch.full((n_curve, n_kept), torch.nan)
        roc_tpr      = torch.full((n_curve, n_kept), torch.nan)
        pr_precision = torch.full((n_curve, n_kept), torch.nan)
        pr_recall    = torch.full((n_curve, n_kept), torch.nan)
        roc_auc      = torch.full((n_kept,), torch.nan)
        pr_auc       = torch.full((n_kept,), torch.nan)

    iterator = (
        tqdm(range(n_kept), desc="Generating Curves", leave=False)
        if thresholds_curve is not None
        else range(n_kept)
    )
    for i in iterator:
        valid = ~x[i].isnan() & ~y[i].isnan()
        x_i, y_i = x[i, valid], y[i, valid]
        if x_i.numel() == 0:
            continue

        x_true = (x_i < x_threshold).long().numpy()
        y_pred = (y_i < y_threshold).long().numpy()

        accuracy[i]          = accuracy_score(x_true, y_pred)
        balanced_accuracy[i] = balanced_accuracy_score(x_true, y_pred)
        f1[i]                = f1_score(x_true, y_pred, zero_division=0)
        precision[i]         = precision_score(x_true, y_pred, zero_division=0)
        recall[i]            = recall_score(x_true, y_pred, zero_division=0)

        if thresholds_curve is None:
            continue

        for t, threshold in enumerate(thresholds_curve):
            y_pred_t = (y_i < threshold).long().numpy()
            tp = ((y_pred_t == 1) & (x_true == 1)).sum()
            fp = ((y_pred_t == 1) & (x_true == 0)).sum()
            tn = ((y_pred_t == 0) & (x_true == 0)).sum()
            fn = ((y_pred_t == 0) & (x_true == 1)).sum()
            roc_tpr[t, i]      = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            roc_fpr[t, i]      = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            pr_precision[t, i] = precision_score(x_true, y_pred_t, zero_division=0)
            pr_recall[t, i]    = recall_score(x_true, y_pred_t, zero_division=0)

        order      = roc_fpr[:, i].argsort()
        roc_auc[i] = float(auc(roc_fpr[order, i], roc_tpr[order, i]))
        order      = pr_recall[:, i].argsort()
        pr_auc[i]  = float(auc(pr_recall[order, i], pr_precision[order, i]))

    out = {
        "accuracy":          accuracy.reshape(kept_shape),
        "balanced_accuracy": balanced_accuracy.reshape(kept_shape),
        "f1":                f1.reshape(kept_shape),
        "precision":         precision.reshape(kept_shape),
        "recall":            recall.reshape(kept_shape),
    }
    if thresholds_curve is not None:
        out |= {
            "roc_fpr":      roc_fpr.reshape(n_curve, *kept_shape),
            "roc_tpr":      roc_tpr.reshape(n_curve, *kept_shape),
            "roc_auc":      roc_auc.reshape(kept_shape),
            "pr_precision": pr_precision.reshape(n_curve, *kept_shape),
            "pr_recall":    pr_recall.reshape(n_curve, *kept_shape),
            "pr_auc":       pr_auc.reshape(kept_shape),
        }
    return out


def vertically_averaged_oxygen(
        oxygen_data: Tensor,
        measured_depths: Tensor,
        mask: Tensor | None = None
) -> Tensor:

    """Compute the vertically averaged oxygen concentration.
    Input:
        oxygen_data: Tensor of shape (dep, lat, lon) representing oxygen concentration at different depths.
        measured_depths: Tensor of shape (dep,) representing the depths of oxygen data.

    Output:
        Tensor of shape (lat, lon) representing the vertically averaged oxygen concentration.

    """

    # compute the thickness of each layer
    layer_thickness = torch.diff(measured_depths, dim=0, prepend=torch.tensor([0.0], device=measured_depths.device))

    # compute the weighted average of oxygen concentration
    weighted_oxygen = oxygen_data[mask] * layer_thickness[:, None, None]
    vertically_averaged_oxygen = torch.sum(weighted_oxygen, dim=0)

    return vertically_averaged_oxygen


def thickness_bottom_hypoxia(
        oxygen_data: Tensor,
        measured_depths: Tensor,
        mask: Tensor | None = None,
        threshold: float = 63.0
) -> Tensor:
    """Compute the thickness of the bottom hypoxic layer.
    Input:
        oxygen_data: Tensor of shape (dep, lat, lon) representing oxygen concentration at different depths.
        measured_depths: Tensor of shape (dep,) representing the depths of oxygen data.
        threshold: Oxygen concentration threshold for hypoxia (default is 63.0).

    Output:
        Tensor of shape (lat, lon) representing the thickness of the bottom hypoxic layer.

    """
    ### check this function, it is not correct, it should be the thickness of the BOTTOM hypoxic layer, not the total thickness of hypoxic layers

    # Create a mask for hypoxic layers
    hypoxic_mask = oxygen_data < threshold

    general_mask = torch.ones_like(oxygen_data, dtype=torch.bool, device=oxygen_data.device)
    if mask is not None:
        general_mask = general_mask & mask

    # Compute the thickness of each layer
    layer_thickness = torch.diff(measured_depths, dim=0, prepend=torch.tensor([0.0], device=measured_depths.device))

    # From the hypoxic mask and layer thickness, compute the thickness of the bottom hypoxic layer
    bottom_hypoxic_thickness = torch.zeros_like(oxygen_data[0], device=oxygen_data.device)
    for i in range(oxygen_data.shape[0]):
        bottom_hypoxic_thickness += hypoxic_mask[i] * layer_thickness[i]


    return bottom_hypoxic_thickness


def oxygen_partial_pressure(
        oxygen_data: Tensor,
        salinity_data: Tensor,
        temperature_data: Tensor,
) -> Tensor:
    """Compute the partial pressure of oxygen (pO2) from oxygen concentration.
    Input:
        oxygen_data: Tensor of shape (dep, lat, lon) representing oxygen concentration at different depths.
        salinity_data: Tensor of shape (dep, lat, lon) representing salinity at different depths.
        temperature_data: Tensor of shape (dep, lat, lon) representing temperature at different depths.

    Output:
        Tensor of shape (dep, lat, lon) representing the partial pressure of oxygen.

    """

    # compute Henry's Law constant from temperature and salinity
    henry_constant = 0.0001 * (1 + 0.032 * salinity_data) * torch.exp(-0.02 * temperature_data)  ## varify this formula

    po2 = oxygen_data / henry_constant

    return po2


def oxygen_saturation(
        temperature_data: Tensor,
        salinity_data: Tensor,
        pressure_data: Tensor
) -> Tensor:
    """Compute the oxygen saturation from oxygen concentration.
    Input:
        temperature_data: Tensor of shape (dep, lat, lon) representing temperature at different depths.
        salinity_data: Tensor of shape (dep, lat, lon) representing salinity at different depths.
        pressure_data: Tensor of shape (dep, lat, lon) representing pressure at different depths.
    Output:
        Tensor of shape (dep, lat, lon) representing the oxygen saturation.
    """

    ## to check
    o2_saturated = 14.6 * torch.exp(-0.02 * temperature_data) * (1 - 0.032 * salinity_data) * torch.exp(-0.0001 * pressure_data)  ## varify this formula

    return o2_saturated


def oxygen_solubility(
        oxygen_data: Tensor,
        temperature_data: Tensor,
        salinity_data: Tensor,
        pressure_data: Tensor
) -> Tensor:
    """Compute the oxygen solubility from oxygen concentration, temperature and salinity, and pressure.
    Input:
        oxygen_data: Tensor of shape (dep, lat, lon) representing oxygen concentration at different depths.
        temperature_data: Tensor of shape (dep, lat, lon) representing temperature at different depths.
        salinity_data: Tensor of shape (dep, lat, lon) representing salinity at different depths.
        pressure_data: Tensor of shape (dep, lat, lon) representing pressure at different depths.
    Output:
        Tensor of shape (dep, lat, lon) representing the oxygen solubility.
    """

    o2_saturation = oxygen_saturation(temperature_data, salinity_data, pressure_data)
    solubility = oxygen_data / o2_saturation
    return solubility

def apparent_oxygen_utilization(
        oxygen_data: Tensor,
        temperature_data: Tensor,
        salinity_data: Tensor,
        pressure_data: Tensor
) -> Tensor:
    """Compute the apparent oxygen utilization (AOU) from oxygen concentration.
    Input:
        oxygen_data: Tensor of shape (dep, lat, lon) representing oxygen concentration at different depths.
        temperature_data: Tensor of shape (dep, lat, lon) representing temperature at different depths.
        salinity_data: Tensor of shape (dep, lat, lon) representing salinity at different depths.
        pressure_data: Tensor of shape (dep, lat, lon) representing pressure at different depths.
    Output:
        Tensor of shape (dep, lat, lon) representing the apparent oxygen utilization.
    """

    o2_saturation = oxygen_saturation(temperature_data, salinity_data, pressure_data)
    aou = o2_saturation - oxygen_data
    return aou
