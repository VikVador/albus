import torch

from albus.metrics import *


def test_mean_square_error_manual() -> None:
    r"""Matches a hand-computed mean square error over a small tensor."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[2.0, 2.0], [3.0, 6.0]])
    mse = mean_square_error(x, y, dims="T C", reduce="C")
    assert torch.allclose(mse, torch.tensor([0.5, 2.0]))


def test_mean_square_error_ignores_masked_entries() -> None:
    r"""Excludes masked-out entries from the averaged error."""
    x = torch.tensor([1.0, 2.0, 100.0])
    y = torch.tensor([1.0, 4.0, -100.0])
    mask = torch.tensor([1, 1, 0])
    mse = mean_square_error(x, y, dims="X", reduce="X", mask=mask)
    assert torch.isclose(mse, torch.tensor(2.0))


def test_root_mean_square_error_is_sqrt_of_mse() -> None:
    r"""Is the elementwise square root of the mean square error."""
    x, y = torch.randn(4, 5), torch.randn(4, 5)
    mse = mean_square_error(x, y, dims="T X", reduce="X")
    rmse = root_mean_square_error(x, y, dims="T X", reduce="X")
    assert torch.allclose(rmse, mse.sqrt())


def test_standard_deviation_matches_population_std() -> None:
    r"""Matches torch's population (biased) standard deviation."""
    x = torch.randn(3, 100)
    std = standard_deviation(x, dims="T X", reduce="X")
    expected = x.std(dim=1, unbiased=False)
    assert torch.allclose(std, expected, atol=1e-5)


def test_skill_perfect_ensemble() -> None:
    r"""Is zero when every ensemble member equals the ground truth."""
    x = torch.randn(5, 3)
    y = x.unsqueeze(1).expand(5, 4, 3).clone()  # every member equals the truth
    s = skill(x, y, dims="T N C", ensemble="N", reduce="C")
    assert torch.allclose(s, torch.zeros(5), atol=1e-6)


def test_spread_zero_when_ensemble_is_constant() -> None:
    r"""Is zero when the ensemble has no variability across its members."""
    y = torch.ones(2, 6, 3)  # no variability across the ensemble axis
    sp = spread(y, dims="T N C", ensemble="N", reduce="C")
    assert torch.allclose(sp, torch.zeros(2), atol=1e-6)


def test_spread_matches_unbiased_torch_var() -> None:
    r"""Matches torch's unbiased ensemble variance, averaged spatially and square-rooted."""
    y = torch.randn(2, 8, 5)
    sp = spread(y, dims="T N C", ensemble="N", reduce="C")
    expected = y.var(dim=1, unbiased=True).mean(dim=-1).sqrt()
    assert torch.allclose(sp, expected, atol=1e-5)


def test_spread_skill_ratio_matches_individual_calls() -> None:
    r"""Returns the same skill and spread as calling those functions separately."""
    x = torch.randn(4, 5)
    y = torch.randn(4, 6, 5)
    s, sp, ratio = spread_skill_ratio(x, y, dims="T N C", ensemble="N", reduce="C")
    assert torch.allclose(s, skill(x, y, dims="T N C", ensemble="N", reduce="C"))
    assert torch.allclose(sp, spread(y, dims="T N C", ensemble="N", reduce="C"))
    n = y.shape[1]
    assert torch.allclose(ratio, (((n + 1) / n) ** 0.5) * sp / s.clamp(min=1e-8))


def test_crps_matches_brute_force() -> None:
    r"""Matches a brute-force double-loop implementation of the fair CRPS formula."""
    torch.manual_seed(0)
    x = torch.randn(3, 2)
    y = torch.randn(3, 6, 2)

    def brute_force(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        T, N, C = y.shape
        out = torch.zeros(T, C)
        for t in range(T):
            for c in range(C):
                members, truth = y[t, :, c], x[t, c]
                term1 = (members - truth).abs().mean()
                term2 = sum((members[i] - members[j]).abs() for i in range(N) for j in range(N))
                out[t, c] = term1 - term2 / (2 * N * (N - 1))
        return out

    # Reduce over a dummy singleton axis, so C is preserved for a direct comparison.
    crps = continuous_ranked_probability_score(
        x.unsqueeze(-1), y.unsqueeze(-1), dims="T N C Z", ensemble="N", reduce="Z"
    )
    assert torch.allclose(crps, brute_force(x, y), atol=1e-5)


def test_crps_zero_for_perfect_constant_ensemble() -> None:
    r"""Is zero when every ensemble member equals the ground truth."""
    x = torch.randn(3, 2)
    y = x.unsqueeze(1).expand(3, 5, 2).clone()
    crps = continuous_ranked_probability_score(x, y, dims="T N C", ensemble="N", reduce="C")
    assert torch.allclose(crps, torch.zeros(3), atol=1e-6)


def _reference_power_spectrum(u: torch.Tensor) -> torch.Tensor:
    fft = torch.fft.rfft2(u, norm="ortho")
    psd = fft.real.pow(2) + fft.imag.pow(2)
    ny, nx = u.shape[-2], u.shape[-1]
    ky = torch.fft.fftfreq(ny) * ny
    kx = torch.fft.rfftfreq(nx) * nx
    k_grid = torch.round((ky[:, None] ** 2 + kx[None, :] ** 2).sqrt()).long()
    n_bins = min(ny, nx) // 2 + 1
    k_flat = k_grid.clamp(0, n_bins - 1).reshape(-1)
    count = torch.zeros(n_bins)
    count.scatter_add_(0, k_flat, torch.ones_like(k_flat, dtype=count.dtype))
    count = count.clamp(min=1)
    batch = psd.shape[:-2]
    spectrum = torch.zeros(*batch, n_bins)
    spectrum.scatter_add_(-1, k_flat.expand(*batch, -1), psd.reshape(*batch, -1))
    return spectrum / count


def test_power_spectrum_matches_reference_without_mask() -> None:
    r"""Matches a direct port of the original neptune-m1.py power spectrum algorithm."""
    torch.manual_seed(0)
    u = torch.randn(2, 3, 16, 20)
    u = u - u.mean(dim=(-2, -1), keepdim=True)  # our function detrends, so match it here
    spectrum = power_spectrum(u, dims="B C Y X", spatial="Y X")
    assert torch.allclose(spectrum, _reference_power_spectrum(u), atol=1e-4)


def test_power_spectrum_is_axis_order_invariant() -> None:
    r"""Gives the same spectrum regardless of where the spatial axes sit in `dims`."""
    torch.manual_seed(0)
    u = torch.randn(2, 3, 16, 20)
    spectrum = power_spectrum(u, dims="B C Y X", spatial="Y X")
    u_t = u.transpose(-2, -1)
    spectrum_t = power_spectrum(u_t, dims="B C X Y", spatial="Y X")
    assert torch.allclose(spectrum, spectrum_t, atol=1e-4)


def test_power_spectrum_detrend_is_robust_to_offset() -> None:
    r"""Is unaffected by a large constant offset added to the field, thanks to detrending."""
    torch.manual_seed(0)
    u = torch.randn(2, 3, 16, 20)
    mask = (torch.rand(1, 3, 16, 20) > 0.2).float()
    spectrum = power_spectrum(u, dims="B C Y X", spatial="Y X", mask=mask)
    spectrum_offset = power_spectrum(u + 1000.0, dims="B C Y X", spatial="Y X", mask=mask)
    assert torch.allclose(spectrum, spectrum_offset, atol=1e-2)
