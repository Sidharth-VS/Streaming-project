import numpy as np
import pytest

from approxiot.sampling.reservoir import reservoir_sample


def test_returns_everything_when_stream_smaller_than_reservoir():
    items = list(range(5))
    assert reservoir_sample(items, 10, rng=np.random.default_rng(0)) == items


def test_returns_exact_reservoir_size_when_stream_larger():
    items = list(range(100))
    sample = reservoir_sample(items, 7, rng=np.random.default_rng(1))
    assert len(sample) == 7
    assert len(set(sample)) == 7  # no duplicates


def test_size_zero_and_empty_stream():
    assert reservoir_sample(list(range(10)), 0, rng=np.random.default_rng(2)) == []
    assert reservoir_sample([], 3, rng=np.random.default_rng(3)) == []


def test_negative_size_raises():
    with pytest.raises(ValueError):
        reservoir_sample([1, 2, 3], -1)


def test_same_seed_gives_same_sample():
    items = list(range(50))
    a = reservoir_sample(items, 10, rng=np.random.default_rng(7))
    b = reservoir_sample(items, 10, rng=np.random.default_rng(7))
    assert a == b


def test_uniform_inclusion_probability():
    """Every item must land in the reservoir with probability ~k/n (paper §II-B)."""
    n, k, trials = 60, 12, 3000
    counts = np.zeros(n)
    rng = np.random.default_rng(1234)
    items = list(range(n))
    for _ in range(trials):
        for item in reservoir_sample(items, k, rng=rng):
            counts[item] += 1
    empirical = counts / trials
    expected = k / n
    # 4-sigma binomial band is ~0.06 here; allow a comfortable margin.
    assert np.all(np.abs(empirical - expected) < 0.06), (empirical.min(), empirical.max())
