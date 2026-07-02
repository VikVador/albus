r"""Tests for albus.ocean.oxygen."""

import torch

from albus.ocean.oxygen import hypoxia


def test_hypoxia_perfect_prediction() -> None:
    r"""Gives perfect accuracy and balanced accuracy when the forecast equals the truth."""
    torch.manual_seed(0)
    x = torch.rand(2, 3, 50) * 400
    out = hypoxia(x=x, y=x.clone(), dims="T C N", reduce="N", thresholds=[63.0, 63.0])
    assert torch.allclose(out["accuracy"], torch.ones(2, 3))
    assert torch.allclose(out["balanced_accuracy"].nan_to_num(1.0), torch.ones(2, 3))


def test_hypoxia_output_shapes() -> None:
    r"""Collapses only the `reduce` axes, keeping the other axes in every returned score."""
    x = torch.rand(4, 2, 5, 6) * 400
    y = torch.rand(4, 2, 5, 6) * 400
    out = hypoxia(x=x, y=y, dims="T C Y X", reduce="Y X", thresholds=[63.0, 63.0])
    for key in ("accuracy", "balanced_accuracy", "f1", "precision", "recall"):
        assert out[key].shape == (4, 2)


def test_hypoxia_ignores_masked_entries() -> None:
    r"""Excludes masked-out entries, even when they would otherwise disagree."""
    x = torch.tensor([[100.0, 100.0, 10.0, 10.0]])  # (T=1, N=4)
    y = torch.tensor([[100.0, 10.0, 10.0, 10.0]])  # disagrees at index 1, but it is masked out
    mask = torch.tensor([1, 0, 1, 1])
    out = hypoxia(x=x, y=y, dims="T N", reduce="N", thresholds=[63.0, 63.0], mask=mask)
    assert torch.allclose(out["accuracy"], torch.ones(1))


def test_hypoxia_curve_shapes_and_auc_range() -> None:
    r"""Builds ROC/precision-recall curves of the expected shape, with AUC values in [0, 1]."""
    torch.manual_seed(0)
    x = torch.rand(2, 200) * 400
    y = x + torch.randn(2, 200) * 20
    thresholds_curve = [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0]
    out = hypoxia(
        x=x,
        y=y,
        dims="T N",
        reduce="N",
        thresholds=[63.0, 63.0],
        thresholds_curve=thresholds_curve,
    )
    assert out["roc_fpr"].shape == (len(thresholds_curve), 2)
    assert out["roc_tpr"].shape == (len(thresholds_curve), 2)
    assert out["pr_precision"].shape == (len(thresholds_curve), 2)
    assert out["pr_recall"].shape == (len(thresholds_curve), 2)
    assert ((out["roc_auc"] >= 0) & (out["roc_auc"] <= 1)).all()
    assert ((out["pr_auc"] >= 0) & (out["pr_auc"] <= 1)).all()
