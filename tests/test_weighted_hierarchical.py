import numpy as np

from approxiot.items import Item
from approxiot.sampling.weighted_hierarchical import get_sample_size, whsamp


def _items(values, substream="A1", src_window=0):
    return [Item(substream, float(v), src_window) for v in values]


# --------------------------------------------------------------------------
# getSampleSize (Algorithm 2 line 7)
# --------------------------------------------------------------------------
def test_proportional_allocation_respects_budget():
    counts = {"A1": 100, "B1": 50, "C1": 25, "D1": 5}
    allocation = get_sample_size(60, counts)
    assert sum(allocation.values()) <= 60
    # Proportional shares are honored within rounding.
    assert allocation["A1"] > allocation["B1"] > allocation["C1"] > allocation["D1"]


def test_small_substream_never_neglected():
    """The paper's core promise: no sub-stream may be overlooked."""
    counts = {"A1": 1_000_000, "D1": 1}
    allocation = get_sample_size(10, counts)
    assert allocation["D1"] >= 1
    assert sum(allocation.values()) <= 10


def test_budget_below_number_of_substreams_still_covers_all():
    counts = {"a": 10, "b": 10, "c": 10}
    allocation = get_sample_size(2, counts)
    assert all(v >= 1 for v in allocation.values())
    assert sum(allocation.values()) == 3  # documented min-1 deviation


def test_empty_counts_gives_empty_allocation():
    assert get_sample_size(10, {}) == {}


# --------------------------------------------------------------------------
# Paper Figure 2 worked example: node A samples 3 of 6 items -> w = 2, c = 3
# --------------------------------------------------------------------------
def test_figure2_worked_example():
    result = whsamp(_items([1, 2, 3, 4, 5, 6]), sample_size=3, rng=np.random.default_rng(0))
    assert len(result.sample["A1"]) == 3
    assert result.W_out["A1"] == 2.0  # w = c/N = 6/3, W_in = 1
    assert result.C_out["A1"] == 3
    # The weighted estimate of the full sum is centered on 21 (1+...+6).
    estimate = sum(i.value for i in result.sample["A1"]) * result.W_out["A1"]
    assert 3 <= estimate <= 33  # 3 sampled values, each <= 6, weight 2


def test_weight_multiplies_incoming_weight():
    """W_out = W_in * w_i (Algorithm 2 line 14)."""
    result = whsamp(_items(range(8)), sample_size=4, W_in={"A1": 2.0}, rng=np.random.default_rng(1))
    assert result.W_out["A1"] == 4.0  # 2.0 * (8/4)
    assert result.C_out["A1"] == 4


def test_no_weight_change_when_everything_kept():
    """c_i <= N_i -> W_out = W_in and C_out = c_i (Algorithm 2 lines 17-19)."""
    result = whsamp(_items([5, 6, 7]), sample_size=10, W_in={"A1": 3.0}, rng=np.random.default_rng(2))
    assert result.W_out["A1"] == 3.0
    assert result.C_out["A1"] == 3
    assert len(result.sample["A1"]) == 3


def test_all_substreams_present_in_sample():
    items = _items(range(50), "A1") + _items(range(5), "D1")
    result = whsamp(items, sample_size=50, rng=np.random.default_rng(3))
    assert set(result.sample) == {"A1", "D1"}
    # Proportional quota for D1 is 50*5/55 = 4.5 -> largest-remainder rounds it
    # up to 5, so the small sub-stream is kept in full with unit weight.
    assert len(result.sample["D1"]) == 5
    assert result.W_out["D1"] == 1.0
    # ... while the large one is sampled down and re-weighted.
    assert result.W_out["A1"] == 50 / 45


# --------------------------------------------------------------------------
# Paper Figure 4 / Eq. 9: asynchronous interval calibration
# --------------------------------------------------------------------------
def test_eq9_async_correction_matches_figure4():
    """Paper-verbatim Eq. 9: a 3-node chain must recover W_2^out = c_src / N_2.

    Node 1: c_src=20, N_1=5  -> W_1 = 4, C_1^out = 5.
    Node 2: only alpha*C_in items arrive in its interval; the Eq. 9 factor
    C^in/c must undo the under-estimation for every straddle fraction.
    """
    c_src, n1, n2 = 20, 5, 1

    # Node 1 (source): full window.
    node1 = whsamp(_items(range(c_src)), sample_size=n1, rng=np.random.default_rng(4))
    assert node1.W_out["A1"] == c_src / n1
    C_in = node1.C_out["A1"]  # 5 sampled items sent downstream

    # Node 2 receives the whole downstream sample in one interval (alpha = 1).
    whole = whsamp(
        _items(range(C_in)),
        sample_size=n2,
        W_in=node1.W_out,
        C_in={"A1": C_in},
        rng=np.random.default_rng(5),
        eq9_boost=True,
    )
    assert whole.W_out["A1"] == c_src / n2

    # Node 2 receives only alpha = 0.6 of the downstream sample (straddled items).
    alpha = 0.6
    straddled = whsamp(
        _items(range(int(alpha * C_in))),  # 3 items arrive now
        sample_size=n2,
        W_in=node1.W_out,
        C_in={"A1": C_in},  # metadata still describes all 5
        rng=np.random.default_rng(6),
        eq9_boost=True,
    )
    assert straddled.W_out["A1"] == c_src / n2
    # And the (1 - alpha) remainder arriving next interval yields the same weight
    # (paper Fig. 4: "For (1 - alpha) N_1 items, node 2 uses the saved W_in/C_in").
    remainder = whsamp(
        _items(range(int(alpha * C_in), C_in)),  # 2 items arrive next interval
        sample_size=n2,
        W_in=node1.W_out,
        C_in={"A1": C_in},
        rng=np.random.default_rng(7),
        eq9_boost=True,
    )
    assert remainder.W_out["A1"] == c_src / n2


def test_per_item_weights_keep_straddled_batches_unbiased():
    """Default path: items straddling windows carry per-item effective weights,
    so no Eq. 9-style correction is needed and no batch is double-counted."""
    items = _items(range(10), src_window=0)
    first = whsamp(items, sample_size=5, rng=np.random.default_rng(10))
    assert all(it.eff_weight == 2.0 for it in first.sample["A1"])

    # A downstream node that keeps everything must preserve the weights.
    passed = whsamp(first.sample["A1"], sample_size=100, rng=np.random.default_rng(11))
    assert all(it.eff_weight == 2.0 for it in passed.sample["A1"])

    # ... and the weighted estimate is exactly unbiased: with 10 items of value
    # 10, whatever 5 the reservoir keeps, weight 2 reconstructs the true sum 100.
    census_items = _items([10.0] * 10, src_window=0)
    estimates = []
    for seed in range(20):
        upstream = whsamp(census_items, sample_size=5, rng=np.random.default_rng(100 + seed))
        downstream = whsamp(upstream.sample["A1"], sample_size=100, rng=np.random.default_rng(seed))
        estimates.append(sum(it.value * it.eff_weight for it in downstream.sample["A1"]))
    assert all(e == 100.0 for e in estimates)


def test_async_without_metadata_falls_back_to_sync_weight():
    result = whsamp(
        _items(range(10)),
        sample_size=5,
        async_correction=True,  # no C_in available
        rng=np.random.default_rng(8),
    )
    assert result.W_out["A1"] == 2.0  # 1 * (10/5) * (10/10)


def test_sync_mode_ignores_cin_mismatch():
    """Without the async flag, Eq. 9 is not applied (Algorithm 2 verbatim)."""
    result = whsamp(
        _items(range(3)),
        sample_size=2,
        W_in={"A1": 4.0},
        C_in={"A1": 5},
        rng=np.random.default_rng(9),
        async_correction=False,
    )
    assert result.W_out["A1"] == 4.0 * (3 / 2)
