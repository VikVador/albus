r"""Tests for albus.tools."""

import torch

from albus.tools import *


def test_axes_basic() -> None:
    r"""Resolves each requested axis name to its position in `dims`."""
    assert axes("T N C Y X", "Y X") == (3, 4)


def test_axes_single() -> None:
    r"""Resolves a single axis name to a one-element tuple."""
    assert axes("T N C Y X", "N") == (1,)


def test_axes_reordered() -> None:
    r"""Resolution follows the order of `names`, not the order of `dims`."""
    assert axes("T N C Y X", "X Y") == (4, 3)


def test_drop() -> None:
    r"""Removes the named axis from a dims string."""
    assert drop("T N C Y X", "N") == "T C Y X"


def test_drop_missing_name_is_noop() -> None:
    r"""Leaves the dims string unchanged when the axis to drop is not present."""
    assert drop("T C Y X", "N") == "T C Y X"


def test_apply_mask_default_nan() -> None:
    r"""Replaces masked-out entries with NaN by default, leaving the rest untouched."""
    x = torch.ones(4)
    mask = torch.tensor([1, 1, 0, 0])
    out = apply_mask(x, mask)
    assert torch.equal(out[:2], torch.ones(2))
    assert torch.isnan(out[2:]).all()


def test_apply_mask_custom_value() -> None:
    r"""Replaces masked-out entries with a caller-supplied value instead of NaN."""
    x = torch.ones(4)
    mask = torch.tensor([1, 0, 1, 0])
    out = apply_mask(x, mask, value=0.0)
    assert torch.equal(out, torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_apply_mask_none_is_identity() -> None:
    r"""Returns the input tensor unchanged when no mask is given."""
    x = torch.randn(3, 3)
    assert torch.equal(apply_mask(x, None), x)
