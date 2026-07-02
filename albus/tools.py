r"""Helper tools for metric implementations."""

__all__ = [
    "apply_mask",
    "axes",
    "drop",
]

from torch import Tensor


def apply_mask(
    x: Tensor,
    mask: Tensor | None,
    value: float = float("nan"),
) -> Tensor:
    r"""Replace masked-out entries with a specified value.

    Arguments:
        x     : Input tensor.
        mask  : Boolean-like tensor, broadcastable with `x`.
        value : Value to replace masked-out entries with (default: NaN).

    Returns:
        Tensor with masked-out entries replaced by `value`.
    """
    return x if mask is None else x.masked_fill(mask == 0, value)


def axes(
    dims: str,
    names: str,
) -> tuple[int, ...]:
    r"""Resolve named axes to their positional indices.

    Arguments:
        dims  : Space-separated axis names describing a tensor's shape, e.g. "T N C Y X".
        names : Space-separated subset of `dims` to resolve, e.g. "Y X".

    Returns:
        Positional indices of `names` within `dims`.
    """
    positions = {name: axis for axis, name in enumerate(dims.split())}
    return tuple(positions[name] for name in names.split())


def drop(
    dims: str,
    name: str,
) -> str:
    r"""Remove a named axis from a dims string.

    Arguments:
        dims : Space-separated axis names describing a tensor's shape, e.g. "T N C Y X".
        name : Axis name to remove, e.g. "N".

    Returns:
        `dims` without `name`, e.g. "T C Y X".
    """
    return " ".join(axis for axis in dims.split() if axis != name)
