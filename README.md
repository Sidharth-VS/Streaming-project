# Streaming-project — ApproxIoT replication + adaptive sampling-budget controller

A Python replication of the core system from Wen et al., *"Approximate Edge
Analytics for the IoT Ecosystem"* (ApproxIoT, arXiv:1805.05674), extended with a
novel **closed-loop adaptive sampling-budget controller** that automatically
tunes each node's sampling fraction to meet a user-specified error target —
the manual-tuning step the original paper lists as future work.

Two deliverables:

1. **Replication**: a faithful, smaller-scale reimplementation of ApproxIoT's
   weighted hierarchical sampling (Algorithm 2) reproducing the paper's
   qualitative claims (ApproxIoT beats coin-flip SRS on accuracy at equal
   sampling fraction; savings/sampling trade-off comparable; the gap widens
   under extreme sub-stream skew).
2. **Novelty**: an adaptive controller (P / PID in log-fraction space / a
   bandit) that observes the root's relative standard error and adjusts the
   system-wide sampling fraction in a closed feedback loop over a dedicated
   control plane.

## What maps to what (paper ↔ code)

| Paper concept | Here |
|---|---|
| Algorithm 1 (per-node workflow) | `src/approxiot/node/base_node.py` |
| Algorithm 2 (WHSamp) + Eq. 1 / Eq. 9 | `src/approxiot/sampling/weighted_hierarchical.py` |
| Reservoir sampling (§II-B) | `src/approxiot/sampling/reservoir.py` |
| Sub-stream stratification (§III-B) | `src/approxiot/sampling/stratifier.py` |
| Linear queries (Eq. 2–3, 13) | `src/approxiot/queries.py` |
| Error estimation (Eq. 10–14, 68-95-99.7) | `src/approxiot/error_estimation/error_bounds.py` |
| SRS baseline (§IV-B) | `src/approxiot/baselines/srs.py` |
| Kafka topics (Fig. 5) | `src/approxiot/kafka_io/` (pluggable `InMemoryTransport` default) |
| 25-machine testbed (§V-A) | `src/approxiot/simulation.py` (in-process 8→4→2→1 tree) |
| §V-D/E synthetic workloads | `src/approxiot/datagen/synthetic.py` |
| NYC TLC + Brasov/OpenAQ datasets (§VI) | `src/approxiot/datagen/real_world.py` |
| **Novel**: adaptive budget controller | `src/approxiot/controller/` |

## Architecture

```
 8 sources ──(20 ms)──▶ 4 L1 edge ──(40 ms)──▶ 2 L2 edge ──(80 ms)──▶ 1 root
(max: paper's exact tree shape; window-scheduled in-process transport)

each node:   items ─▶ stratify per sub-stream ─▶ allocate reservoir slots
             ─▶ reservoir-sample ─▶ attach effective weights (W·c/N)
             ─▶ forward {sample, W_out, C_out} to parent
root:        same sampling ─▶ weighted query (sum/mean) ─▶ error bound
             ─▶ (novelty) relative stderr ─▶ controller ─▶ control topic
             ─▶ every node adopts the new fraction
```

## Install / test / run

```sh
pip install -e ".[dev,realworld]"   # numpy, pandas, matplotlib, pytest
python -m pytest                    # 50 unit/integration tests
python -m experiments.run_all       # full experiment suite → results/
# or: make experiments               (make not available on some Windows shells)
```

Optional heavier paths: `pip install -e ".[kafka]"` for the confluent-kafka
transport module; `download=True` arguments in `datagen/real_world.py` fetch
the real NYC TLC/OpenAQ datasets (offline falls back to documented
distribution-matched proxies).

## Experiments → paper figures

| File | Reproduces / studies |
|---|---|
| `experiments/exp1_sampling_fraction.py` | §V-B Figs. 6–9 accuracy/throughput vs sampling fraction |
| `experiments/exp2_window_size.py` | §V-C Fig. 10 e2e latency vs window size |
| `experiments/exp3_fluctuating_rates.py` | §V-D fluctuating rate settings 1/2/3 |
| `experiments/exp4_skew.py` | §V-E extreme-skew workload (the dramatic gap) |
| `experiments/exp5_real_world.py` | §VI real-world datasets (proxy fallback documented) |
| `experiments/exp6_adaptive_controller.py` | **Novel**: static vs adaptive budget under steady / step-change / skew, two error targets |

Each experiment writes per-window rows (CSV), a PNG figure, and a JSON verdict
of the expected *qualitative* paper trend into [results/](results/).
`results/summary.json` aggregates all six verdicts. Regenerate with
`python -m experiments.run_all`.

## The novelty: how the controller works

Every `cadence_windows` windows the root feeds its observed relative standard
error into `AdaptiveBudgetController` (`controller/adaptive_budget.py`):

- **signal**: `sqrt(Var) / |result|` — online, needs no ground truth;
- **stability**: trailing median filter + dead-band around the target +
  minimum dwell time + confirmation-before-actuation (anti-thrashing);
- **policies** (`controller/policies.py`): A. proportional P-controller;
  B. PID in log-fraction space with anti-windup; C. ε-greedy bandit (stretch);
- **propagation**: a dedicated control topic broadcasts `ControlMessage`s to
  every node (control never piggybacks on the data plane);
- budgets take effect from the next window onward.

EXP6 evaluates static-tuned / static-conservative / adaptive-P / adaptive-PID
under (a) steady state, (b) a mid-run rate step-change, (c) extreme skew —
reporting bandwidth saved and error-target violations per regime.

## Differences from the original paper

- **Stack**: Java + Kafka Streams on 25 physical machines → Python 3.10+,
  pure in-process simulation with a window-scheduled bus standing in for
  Kafka and machine deployment (a `confluent-kafka`-backed transport is
  provided for the real-broker path but the experiments use the in-process
  bus for reproducibility).
- **Latency model**: `tc-netem`-emulated 20/40/80 ms WAN latencies become
  message arrival-window scheduling (plus an optional item-straddling jitter
  mode that exercises the paper's asynchronous-interval Eq. 9 calibration).
- **Scale/absolute rates**: the paper's 10⁴–10⁵ items/sec workloads are scaled
  down (`experiments/common.py:SIM_RATE_SCALE`) so single runs take seconds;
  relative rates, distributions, and the tree shape are preserved, so all
  qualitative trends carry over.
- **Datasets**: NYC TLC trips and Brasov air-quality readings are substituted
  with (a) TLC Parquet files fetched on demand / (b) the OpenAQ v2 API; when
  the network is unavailable, distribution-matched proxies (log-normal
  amounts / pollutant readings) are used and the `proxy_used` flag is recorded
  in every EXP5 output.
- **Exact numbers will differ** from the paper's figures; the replication
  targets the same qualitative patterns, which the experiment suite checks
  programmatically and reports in `results/summary.json`.

## Layout

```
src/approxiot/     implementation (see table above)
experiments/       exp1..exp6 + run_all
tests/             50 tests covering algorithms, bounds, SRS, controller, sim
results/           CSVs / PNGs / JSON verdicts (gitignored except .gitkeep)
notebooks/         results_analysis.ipynb — loads results/ and redraws figures
```
