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
| 25-machine testbed (§V-A) | `src/approxiot/simulation.py` (in-process) + `docker-compose.yml` / `runtime/` (real Redpanda) |
| WAN latency (§V-A) | `tc-netem` 20/40/80 ms in Docker; `src/approxiot/latency.py` event trace for experiments |
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
python -m pytest                    # 63 unit/integration tests
python -m experiments.run_all       # full experiment suite → results/
# or: make experiments               (make not available on some Windows shells)

# Containerized deployment (paper's Fig. 5 topology on a real broker):
docker compose up -d --build         # Redpanda + 15 nodes, 60 windows
MAX_WINDOWS=20 docker compose up -d --build   # short smoke run
docker compose logs -f root          # watch the root (Ctrl-C stops following)
docker compose ps                    # nodes exit after their windows
cat results/docker_results.csv       # estimate, error bound, fraction, latency
docker compose down                  # stop the broker when done
```

The deployment path builds one image (`Dockerfile`) and starts 8 sources →
4 L1 → 2 L2 → 1 root with a Redpanda broker (`docker-compose.yml`).  Every
container runs `python -m approxiot.runtime.node`; `scripts/docker-entrypoint.sh`
applies the paper's 20/40/80 ms `tc-netem` egress latency per tier.  Workload,
rate scale, window size, sampling fraction, query and controller target are
environment variables in the compose file.

Optional heavier paths: `pip install -e ".[kafka]"` for the confluent-kafka
transport module (used by the deployment path); `download=True` arguments in
`datagen/real_world.py` fetch the real NYC TLC/OpenAQ datasets (offline falls
back to documented distribution-matched proxies).

See [RUNNING.md](RUNNING.md) for the full command reference, verification log,
and the current status of the adaptive-controller novelty.

## Experiments → paper figures

| File | Reproduces / studies |
|---|---|
| `experiments/exp1_sampling_fraction.py` | §V-B Figs. 6–9 accuracy/throughput vs fraction, Gaussian **and Poisson** |
| `experiments/exp2_window_size.py` | §V-C Fig. 10 measured latency vs window, ApproxIoT/SRS/native |
| `experiments/exp3_fluctuating_rates.py` | §V-D rate settings 1/2/3, Gaussian **and Poisson** (Fig. 11a/b) |
| `experiments/exp4_skew.py` | §V-E extreme-skew workload (the dramatic gap) |
| `experiments/exp5_real_world.py` | §VI real-world datasets (proxy fallback documented) |
| `experiments/exp6_adaptive_controller.py` | **Novel**: static vs adaptive budget under steady / step-change / skew, two error targets |

Each experiment writes its CSV output, a PNG figure, and (where it defines
one) a JSON verdict of the expected *qualitative* paper trend into
[results/](results/). `results/summary.json` aggregates all six verdicts.
Regenerate with `python -m experiments.run_all`.

**Metrics that are measured, not closed-form.** Throughput is the number of
items processed per *simulated* second by the input-saturated cluster (§V-B),
not Python wall-clock speed, so it does not grow with the window size.  EXP1/EXP2
use `PAPER_RATE_SCALE` ≈ 11k items/s — the paper's measured operating point.
Latency comes from an event trace (`src/approxiot/latency.py`): each item is
created at a time inside its source window, waits for window boundaries
(ApproxIoT/native), spends `n / capacity` seconds at each node and crosses the
20/40/80 ms links.  **SRS is windowless** per §V-C, so its latency stays flat as
the window grows while ApproxIoT's increases.

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
(adaptive policies at both 0.5% and 0.1% targets) under (a) steady state,
(b) a mid-run rate step-change, (c) extreme skew — reporting bandwidth saved,
error-target violations, fraction stability and time-to-reconverge per regime.

## Differences from the original paper

- **Stack**: Java + Kafka Streams on 25 physical machines → Python 3.10+,
  pure in-process simulation with a window-scheduled bus standing in for
  Kafka and machine deployment (a `confluent-kafka`-backed transport is
  provided for the real-broker path but the experiments use the in-process
  bus for reproducibility).
- **Latency measurement**: 20/40/80 ms `tc-netem` links are applied in the
  Docker deployment; the in-process experiments use an event-trace latency
  model (window waits + per-node processing at capacity), not the closed-form
  `3·window + link` formula.  Absolute latencies therefore differ from the
  paper's cluster but the Fig. 10 trend (ApproxIoT grows with window, SRS flat)
  is measured rather than assumed.
- **Throughput**: reported as simulated items/second (paper §V-B definition).
  EXP1/EXP2 run at the paper's ~11k items/s operating point; EXP3–6 remain at a
  reduced `SIM_RATE_SCALE` for runtime, with relative rates preserved.
- **Datasets**: NYC TLC trips and Brasov air-quality readings are substituted
  with (a) TLC Parquet files fetched on demand / (b) the OpenAQ v2 API; when
  the network is unavailable, distribution-matched proxies (log-normal
  amounts / pollutant readings) are used and the `proxy_used` flag is recorded
  in every EXP5 output.
- **Exact numbers will differ** from the paper's figures; the replication
  targets the same qualitative patterns, which the experiment suite checks
  programmatically and reports in `results/summary.json`.
- **Latency saturation is not modelled.** The event trace includes window
  waits, link delays and per-node processing at 12k items/s, but not CPU queue
  build-up on a saturated cluster.  As a result the measured speed-up of 10%
  sampling over native execution is ~1.2-1.4× at a 1 s window, smaller than the
  paper's 6×; the ordering and the Fig. 10 window-size trend are reproduced.
- **Throughput** is the simulated input rate at the paper's saturated-cluster
  operating point (~11k items/s for EXP1/EXP2).  EXP3–6 run at a reduced rate
  (`SIM_RATE_SCALE`) for runtime, preserving relative rates.

## Layout

```
src/approxiot/     implementation (see table above)
  latency.py       event-trace latency measurement (Figs. 9-10)
  runtime/         container node entry point (Kafka/Redpanda deployment)
experiments/       exp1..exp6 + run_all
tests/             tests covering algorithms, bounds, SRS, controller, sim, runtime
RUNNING.md         run/test command reference + verification log
docker-compose.yml Redpanda + 8/4/2/1 node containers + tc-netem entrypoint
results/           CSVs / PNGs / JSON verdicts (gitignored except .gitkeep)
notebooks/         results_analysis.ipynb — loads results/ and redraws figures
```
