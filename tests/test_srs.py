import numpy as np

from approxiot.baselines.srs import SRSSampler, coin_flip_sample
from approxiot.items import Item


def _items(values, substream="A1", src_window=0):
    return [Item(substream, float(v), src_window) for v in values]


def test_coin_flip_bounds():
    rng = np.random.default_rng(0)
    items = _items(range(100))
    assert coin_flip_sample(items, 1.0, rng) == items
    assert coin_flip_sample(items, 0.0, rng) == []


def test_coin_flip_respects_expected_fraction():
    rng = np.random.default_rng(1)
    items = _items(range(10_000))
    kept = coin_flip_sample(items, 0.3, rng)
    assert abs(len(kept) / len(items) - 0.3) < 0.03


def test_srs_invalid_fraction_raises():
    with np.testing.assert_raises(ValueError):
        SRSSampler(0.0)
    with np.testing.assert_raises(ValueError):
        SRSSampler(1.5)


def test_srs_inverse_probability_weights():
    """Each layer multiplies W by 1/f, so L layers give W = (1/f)^L."""
    rng = np.random.default_rng(2)
    items = _items(range(1000))
    layer1 = SRSSampler(0.5).process_window(items, rng=rng)
    assert layer1.W_out["A1"] == 2.0

    layer2 = SRSSampler(0.5).process_window(
        [Item("A1", 1.0)] * layer1.C_out["A1"], W_in=layer1.W_out, rng=rng
    )
    assert layer2.W_out["A1"] == 4.0


def test_srs_estimator_is_unbiased_on_expectation():
    """Monte-Carlo: mean of SRS estimates converges to the true sum."""
    rng = np.random.default_rng(3)
    true_sum = 5000.0  # 100 items of value 50
    estimates = []
    for trial in range(400):
        sampler = SRSSampler(0.4)
        result = sampler.process_window(
            _items([50.0] * 100), rng=np.random.default_rng(1000 + trial)
        )
        estimates.append(sum(i.value for i in result.sample["A1"]) * result.W_out["A1"])
    mean_estimate = float(np.mean(estimates))
    # Standard error ~ sqrt(100 * 0.24 / 0.16) / sqrt(400) * 50 ~ 6; allow 3 sigma.
    assert abs(mean_estimate - true_sum) < 20.0, mean_estimate


def test_srs_keeps_metadata_contract_with_whsamp():
    """Same output type as whsamp so the root can treat both identically."""
    rng = np.random.default_rng(4)
    items = _items(range(50), "A1") + _items(range(20), "B1")
    result = SRSSampler(0.5).process_window(items, rng=rng)
    assert set(result.sample) <= {"A1", "B1"}
    assert result.W_out["A1"] == 2.0 and result.W_out["B1"] == 2.0
    assert all(isinstance(c, int) for c in result.C_out.values())
