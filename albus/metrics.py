r"""Generic metrics for evaluating learned models of physical systems."""

__all__ = [
    "continuous_ranked_probability_score",
    "mean_square_error",
    "power_spectrum",
    "root_mean_square_error",
    "skill",
    "spread",
    "spread_skill_ratio",
    "standard_deviation",
]

import torch

from math import sqrt
from torch import Tensor

from albus.tools import apply_mask, axes, drop


# fmt: off
#
def mean_square_error(
    x: Tensor,
    y: Tensor,
    dims: str,
    reduce: str,
    mask: Tensor | None = None,
) -> Tensor:
    r"""Compute the mean square error between two tensors, ignoring masked entries.

    Formula:
        MSE = 𝔼[(x - y)²]

    Arguments:
        x      : First tensor.
        y      : Second tensor, broadcastable with `x`.
        dims   : Space-separated axis names describing the shape of `x`, e.g. "T N C Y X".
        reduce : Space-separated subset of `dims` to average over, e.g. "N Y X".
        mask   : Boolean-like tensor, broadcastable with `x`, zero where entries are invalid.

    Returns:
        Mean square error, with the axes in `reduce` collapsed.

    Example:
        >>> x = torch.randn(7, 4, 2, 128, 256)  # (T, N, C, Y, X)
        >>> y = torch.randn(7, 4, 2, 128, 256)  # (T, N, C, Y, X)
        >>> mean_square_error(
        ...     x=x,
        ...     y=y,
        ...     dims="T N C Y X",
        ...     reduce="N Y X",
        ... ).shape
        torch.Size([7, 2])
    """
    ax   = axes(dims, reduce)
    se   = (apply_mask(x, mask) - apply_mask(y, mask)).pow(2)
    mse  = se.nanmean(dim=ax)
    return mse


def root_mean_square_error(
    x: Tensor,
    y: Tensor,
    dims: str,
    reduce: str,
    mask: Tensor | None = None,
) -> Tensor:
    r"""Compute the root mean square error between two tensors, ignoring masked entries.

    Formula:
        RMSE = √(𝔼[(x - y)²])

    Arguments:
        x      : First tensor.
        y      : Second tensor, broadcastable with `x`.
        dims   : Space-separated axis names describing the shape of `x`, e.g. "T N C Y X".
        reduce : Space-separated subset of `dims` to average over, e.g. "N Y X".
        mask   : Boolean-like tensor, broadcastable with `x`, zero where entries are invalid.

    Returns:
        Root mean square error, with the axes in `reduce` collapsed.

    Example:
        >>> x = torch.randn(7, 4, 2, 128, 256)  # (T, N, C, Y, X)
        >>> y = torch.randn(7, 4, 2, 128, 256)  # (T, N, C, Y, X)
        >>> root_mean_square_error(
        ...     x=x,
        ...     y=y,
        ...     dims="T N C Y X",
        ...     reduce="N Y X",
        ... ).shape
        torch.Size([7, 2])
    """
    return mean_square_error(x, y, dims, reduce, mask).sqrt()


def standard_deviation(
    x: Tensor,
    dims: str,
    reduce: str,
    mask: Tensor | None = None,
) -> Tensor:
    r"""Compute the (population) standard deviation of a tensor, ignoring masked entries.

    Formula:
        σ = √(𝔼[(x - 𝔼[x])²])

    Arguments:
        x      : Input tensor.
        dims   : Space-separated axis names describing the shape of `x`, e.g. "T N C Y X".
        reduce : Space-separated subset of `dims` to compute the deviation over, e.g. "N Y X".
        mask   : Boolean-like tensor, broadcastable with `x`, zero where entries are invalid.

    Returns:
        Standard deviation, with the axes in `reduce` collapsed.

    Example:
        >>> x = torch.randn(7, 4, 2, 128, 256)  # (T, N, C, Y, X)
        >>> standard_deviation(
        ...     x=x,
        ...     dims="T N C Y X",
        ...     reduce="N Y X",
        ... ).shape
        torch.Size([7, 2])
    """
    ax       = axes(dims, reduce)
    x        = apply_mask(x, mask)
    mean     = x.nanmean(dim=ax, keepdim=True)
    variance = (x - mean).pow(2).nanmean(dim=ax)
    std      = variance.sqrt()
    return std


def skill(
    x: Tensor,
    y: Tensor,
    dims: str,
    ensemble: str,
    reduce: str,
    mask: Tensor | None = None,
) -> Tensor:
    r"""Compute the skill score, i.e. the RMSE of the ensemble mean against the ground truth.

    Formula:
        Skill = √(𝔼[(x - 𝔼_N[y])²])

    Arguments:
        x        : Ground truth tensor.
        y        : Ensemble tensor, broadcastable with `x` once `ensemble` is collapsed.
        dims     : Space-separated axis names describing the shape of `y`, e.g. "T N C Y X".
        ensemble : Name of the ensemble axis within `dims`, e.g. "N".
        reduce   : Space-separated subset of `dims` to average over, e.g. "Y X".
        mask     : Boolean-like tensor, broadcastable with `y`, zero where entries are invalid.

    Returns:
        Skill score, with the axes in `reduce` collapsed.

    Example:
        >>> x = torch.randn(7, 2, 128, 256)     # (T, C, Y, X)
        >>> y = torch.randn(7, 4, 2, 128, 256)  # (T, N, C, Y, X)
        >>> skill(
        ...     x=x,
        ...     y=y,
        ...     dims="T N C Y X",
        ...     ensemble="N",
        ...     reduce="Y X",
        ... ).shape
        torch.Size([7, 2])
    """
    n_axis = axes(dims, ensemble)[0]
    y_mean = apply_mask(y, mask).nanmean(dim=n_axis)
    skll   = root_mean_square_error(x, y_mean, drop(dims, ensemble), reduce, mask)
    return skll


def spread(
    x: Tensor,
    dims: str,
    ensemble: str,
    reduce: str,
    mask: Tensor | None = None,
) -> Tensor:
    r"""Compute the ensemble spread, i.e. the root mean unbiased ensemble variance.

    Formula:
        Spread = √(𝔼[ 1/(N - 1) · Σₙ (x_n - 𝔼_N[x])² ])

    Arguments:
        x        : Ensemble tensor.
        dims     : Space-separated axis names describing the shape of `x`, e.g. "T N C Y X".
        ensemble : Name of the ensemble axis within `dims`, e.g. "N".
        reduce   : Space-separated subset of `dims` (excluding `ensemble`) to average over, e.g. "Y X".
        mask     : Boolean-like tensor, broadcastable with `x`, zero where entries are invalid.

    Returns:
        Ensemble spread, with the axes in `reduce` collapsed.

    Example:
        >>> x = torch.randn(7, 4, 2, 128, 256)  # (T, N, C, Y, X)
        >>> spread(
        ...     x=x,
        ...     dims="T N C Y X",
        ...     ensemble="N",
        ...     reduce="Y X",
        ... ).shape
        torch.Size([7, 2])
    """
    n_axis       = axes(dims, ensemble)[0]
    n            = x.shape[n_axis]
    x            = apply_mask(x, mask)
    mean         = x.nanmean(dim=n_axis, keepdim=True)
    variance     = (x - mean).pow(2).nanmean(dim=n_axis) * (n / (n - 1))
    spatial_axes = axes(drop(dims, ensemble), reduce)
    sprd         = variance.nanmean(dim=spatial_axes).sqrt()
    return sprd


def spread_skill_ratio(
    x: Tensor,
    y: Tensor,
    dims: str,
    ensemble: str,
    reduce: str,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Compute the skill, the spread, and their ratio (corrected for ensemble size).

    Formula:
        Ratio = √((N + 1) / N) · Spread / Skill

    Arguments:
        x        : Ground truth tensor.
        y        : Ensemble tensor, broadcastable with `x` once `ensemble` is collapsed.
        dims     : Space-separated axis names describing the shape of `y`, e.g. "T N C Y X".
        ensemble : Name of the ensemble axis within `dims`, e.g. "N".
        reduce   : Space-separated subset of `dims` (excluding `ensemble`) to average over, e.g. "Y X".
        mask     : Boolean-like tensor, broadcastable with `y`, zero where entries are invalid.

    Returns:
        skill  : Skill score, with the axes in `reduce` collapsed.
        spread : Ensemble spread, with the axes in `reduce` collapsed.
        ratio  : Spread/skill ratio, with the axes in `reduce` collapsed.

    Example:
        >>> x = torch.randn(7, 2, 128, 256)     # (T, C, Y, X)
        >>> y = torch.randn(7, 4, 2, 128, 256)  # (T, N, C, Y, X)
        >>> skll, sprd, ratio = spread_skill_ratio(
        ...     x=x,
        ...     y=y,
        ...     dims="T N C Y X",
        ...     ensemble="N",
        ...     reduce="Y X",
        ... )
        >>> ratio.shape
        torch.Size([7, 2])
    """
    n     = y.shape[axes(dims, ensemble)[0]]
    skll  = skill(x, y, dims, ensemble, reduce, mask)
    sprd  = spread(y, dims, ensemble, reduce, mask)
    ratio = sqrt((n + 1) / n) * sprd / skll.clamp(min=1e-8)
    return skll, sprd, ratio


def continuous_ranked_probability_score(
    x: Tensor,
    y: Tensor,
    dims: str,
    ensemble: str,
    reduce: str,
    mask: Tensor | None = None,
) -> Tensor:
    r"""Compute the (fair) Continuous Ranked Probability Score (CRPS) of an ensemble.

    Formula:
        CRPS = 𝔼_N[|x - y|] - 1/(2N(N - 1)) · Σᵢ Σⱼ |yᵢ - yⱼ|

    where yᵢ, yⱼ range over the ensemble members along `ensemble`.

    Arguments:
        x        : Ground truth tensor.
        y        : Ensemble tensor, broadcastable with `x` once `ensemble` is collapsed.
        dims     : Space-separated axis names describing the shape of `y`, e.g. "T N C Y X".
        ensemble : Name of the ensemble axis within `dims`, e.g. "N".
        reduce   : Space-separated subset of `dims` (excluding `ensemble`) to average over, e.g. "Y X".
        mask     : Boolean-like tensor, broadcastable with `y`, zero where entries are invalid.

    Returns:
        CRPS, with the axes in `reduce` collapsed.

    Example:
        >>> x = torch.randn(7, 2, 128, 256)     # (T, C, Y, X)
        >>> y = torch.randn(7, 4, 2, 128, 256)  # (T, N, C, Y, X)
        >>> continuous_ranked_probability_score(
        ...     x=x,
        ...     y=y,
        ...     dims="T N C Y X",
        ...     ensemble="N",
        ...     reduce="Y X",
        ... ).shape
        torch.Size([7, 2])
    """
    n_axis             = axes(dims, ensemble)[0]
    x                  = apply_mask(x, mask)
    y                  = apply_mask(y, mask)
    n                  = y.shape[n_axis]
    mae                = (y - x.unsqueeze(n_axis)).abs().nanmean(dim=n_axis)
    y_sorted, _        = y.sort(dim=n_axis)
    rank_shape         = [1] * y.dim()
    rank_shape[n_axis] = n
    rank               = torch.arange(1, n + 1, dtype=y.dtype, device=y.device).view(rank_shape)
    spread_term        = ((2 * rank - n - 1) * y_sorted).sum(dim=n_axis) / (n * (n - 1))
    crps               = mae - spread_term
    spatial_axes       = axes(drop(dims, ensemble), reduce)
    crps               = crps.nanmean(dim=spatial_axes)
    return crps


def power_spectrum(
    x: Tensor,
    dims: str,
    spatial: str,
    mask: Tensor | None = None,
) -> Tensor:
    r"""Compute the isotropic 2D power spectrum of a real field, along two spatial axes.

    Formula:
        P(k) = 𝔼_{‖κ‖ = k}[ |F(x)(κ)|² ]

    where F denotes the 2D discrete Fourier transform over `spatial`, and κ its frequency.

    Arguments:
        x       : Input tensor.
        dims    : Space-separated axis names describing the shape of `x`, e.g. "T C Y X".
        spatial : Space-separated names of the two spatial axes to transform, e.g. "Y X".
        mask    : Boolean-like tensor, broadcastable with `x`, zero where entries are invalid.

    Returns:
        Isotropic power spectrum, with `spatial` collapsed into a single radial frequency axis.

    Example:
        >>> x = torch.randn(7, 2, 128, 256)  # (T, C, Y, X)
        >>> power_spectrum(
        ...     x=x,
        ...     dims="T C Y X",
        ...     spatial="Y X",
        ... ).shape
        torch.Size([7, 2, 65])
    """
    y_axis, x_axis = axes(dims, spatial)
    x              = x.movedim((y_axis, x_axis), (-2, -1))
    ny, nx         = x.shape[-2], x.shape[-1]

    # Used to avoid spectral leakage from non-periodic boundaries
    mean = apply_mask(x, mask).nanmean(dim=(-2, -1), keepdim=True)
    x    = apply_mask(x - mean, mask, value=0.0).nan_to_num(0.0)

    fft = torch.fft.rfft2(x, norm="ortho")
    psd = fft.real.pow(2) + fft.imag.pow(2)

    ky     = torch.fft.fftfreq(ny, device=x.device) * ny
    kx     = torch.fft.rfftfreq(nx, device=x.device) * nx
    k_grid = torch.round((ky[:, None].pow(2) + kx[None, :].pow(2)).sqrt()).long()
    n_bins = min(ny, nx) // 2 + 1
    k_flat = k_grid.clamp(0, n_bins - 1).reshape(-1)

    count = torch.zeros(n_bins, device=x.device)
    count.scatter_add_(0, k_flat, torch.ones_like(k_flat, dtype=count.dtype))
    count = count.clamp(min=1)

    batch    = psd.shape[:-2]
    psd_flat = psd.reshape(*batch, -1)
    k_exp    = k_flat.expand(*batch, -1)
    spectrum = torch.zeros(*batch, n_bins, device=x.device)
    spectrum.scatter_add_(dim=-1, index=k_exp, src=psd_flat)
    spectrum = spectrum / count

    return spectrum
