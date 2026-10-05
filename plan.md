# ApproxIoT Replication + Adaptive Budget Controller — Implementation Plan

## 0. Project Summary

Replicate the core system from Wen et al., "Approximate Edge Analytics for the IoT Ecosystem" (ApproxIoT, arXiv:1805.05674), then extend it with a novel **adaptive sampling-budget controller** that automatically tunes each node's sample size to meet a user-specified error target, removing the manual tuning step the original paper lists as future work.

**Two deliverables:**
1. A faithful, smaller-scale reimplementation of ApproxIoT's weighted hierarchical sampling algorithm, reproducing its qualitative claims (ApproxIoT beats simple random sampling (SRS) on accuracy at equal sampling fraction; throughput is comparable to native/SRS; accuracy gap widens under skewed sub-stream rates).
2. A new adaptive controller module that observes the error-estimation output and adjusts per-node sampling fractions in a closed feedback loop, evaluated against the original's static/manual budget approach.

---

## 1. Scope Decisions (do these first, before writing code)

- **Language/stack**: Python, using **Faust** or **Quix Streams** (Kafka-API stream processing libraries) instead of Java Kafka Streams. Rationale: faster iteration for research code; Kafka wire-protocol compatible so swapping in real Kafka later is trivial. If the agent finds Faust unmaintained, fall back to **Quix Streams** or plain `confluent-kafka-python` with manual topology wiring.
- **Broker**: Kafka via `confluent/cp-kafka` Docker image, OR **Redpanda** (`redpandadata/redpanda` Docker image) — Redpanda is lighter-weight and faster to spin up locally; prefer it unless the agent has reason to need real Kafka-specific features.
- **Topology emulation**: Docker Compose, one container per node, with `tc-netem` applied inside containers (or via a sidecar) to inject the paper's latency figures (20ms source→L1, 40ms L1→L2, 80ms L2→root). Do NOT attempt physical multi-machine deployment.
- **Scale**: 4-layer tree — 8 source nodes → 4 L1 nodes → 2 L2 nodes → 1 root node. Same shape as the paper, just containerized instead of on 25 physical machines.
- **Datasets**:
  - Synthetic: generate Gaussian and Poisson sub-streams in code (parameters given in Section 6 below) — no external dependency.
  - Real-world #1: NYC TLC Trip Record Data (substitute any available month/year; schema-compatible with the paper's 2013 dataset — use `total_amount` field as the payment query target).
  - Real-world #2: substitute Brasov pollution data with **OpenAQ API** historical pollution data (particulate matter, CO, SO2, NO2) for any city, if the original Brasov dataset is unavailable.

---

## 2. Repository Structure

```
approxiot-replication/
├── plan.md                          # this file
├── README.md
├── pyproject.toml
├── docker-compose.yml               # broker + node containers + network emulation
├── src/
│   ├── sampling/
│   │   ├── reservoir.py             # basic reservoir sampling (Algorithm from §II-B)
│   │   ├── weighted_hierarchical.py # Algorithm 2 (WHSamp)
│   │   └── stratifier.py            # sub-stream stratification logic
│   ├── node/
│   │   ├── base_node.py             # Algorithm 1 — per-node workflow loop
│   │   ├── sampling_node.py         # edge/sampling node (non-root)
│   │   └── root_node.py             # root node: runs query + error estimation
│   ├── error_estimation/
│   │   └── error_bounds.py          # §III-D: variance/error estimation, 68-95-99.7 rule
│   ├── controller/                  # <-- NOVELTY MODULE
│   │   ├── adaptive_budget.py       # closed-loop budget controller
│   │   └── policies.py              # controller strategies (see §5)
│   ├── baselines/
│   │   └── srs.py                   # simple random sampling baseline (coin-flip)
│   ├── datagen/
│   │   ├── synthetic.py             # Gaussian/Poisson stream generators
│   │   └── real_world.py            # NYC taxi / pollution dataset loaders
│   ├── kafka_io/
│   │   ├── producer.py
│   │   └── consumer.py
│   └── metrics/
│       └── collector.py             # throughput, accuracy loss, latency, bandwidth
├── experiments/
│   ├── exp1_sampling_fraction.py    # §V-B equivalent
│   ├── exp2_window_size.py          # §V-C equivalent
│   ├── exp3_fluctuating_rates.py    # §V-D equivalent
│   ├── exp4_skew.py                 # §V-E equivalent
│   ├── exp5_real_world.py           # §VI equivalent
│   └── exp6_adaptive_controller.py  # NOVELTY: controller vs static budget
├── notebooks/
│   └── results_analysis.ipynb       # plots: accuracy/throughput/latency vs. baseline paper figures
├── tests/
│   ├── test_reservoir.py
│   ├── test_weighted_hierarchical.py
│   ├── test_error_bounds.py
│   └── test_adaptive_controller.py
└── results/                         # output CSVs/plots land here, gitignored except .gitkeep
```

---

## 3. Phase-by-Phase Implementation Plan

### Phase 1 — Core sampling algorithms (no networking yet, pure unit-testable logic)
1. Implement `reservoir.py`: classic Vitter reservoir sampling, reservoir size `N`, probability `N/i` replacement rule.
2. Implement `stratifier.py`: groups incoming items into sub-streams `S_i` by source key.
3. Implement `weighted_hierarchical.py` — this is **Algorithm 2** from the paper:
   - Input: `items, sampleSize, W_in, C_in`
   - Stratify → determine per-substream reservoir size `N_i` via `getSampleSize` (proportional allocation is the simplest correct choice — document the allocation policy chosen)
   - Run reservoir sampling per sub-stream
   - Compute `w_i = c_i / N_i` (if `c_i > N_i`, else 1), update `W_i_out = W_i_in * w_i`, `C_i_out = c_i`
   - Return `sample, W_out, C_out`
4. Write unit tests reproducing the paper's worked example in Figure 2 (node A samples 3 of 6 items, verify resulting weight/count match the paper's numbers).
5. Implement the asynchronous time-interval correction (§III-C, Equation 9) as an optional mode on the same module — this is required for the multi-node case since container network delay makes synchronized arrival unrealistic.

**Acceptance criteria**: unit tests pass; manually verify weight propagation across a 3-node toy chain matches the paper's Figure 4 worked example (expected `W_i,2_out = c_i,src / N_i,2`).

### Phase 2 — Error estimation module
1. Implement `error_bounds.py`:
   - `estimate_variance_sum()` — Equation 11
   - `estimate_variance_mean()` — Equation 14
   - `error_bound()` — applies 68-95-99.7 rule using sqrt(variance)
2. Unit test against hand-computed small examples.

### Phase 3 — Node workflow (Algorithm 1) + Kafka/Redpanda wiring
1. Implement `base_node.py`: the per-time-interval loop — `costFunction(budget) → size`, consume stream + metadata, call `WHSamp`, forward to parent or (if root) run query + error estimation.
2. `sampling_node.py`: non-root behavior — subscribe to upstream topic, publish sampled output + `W_out`/`C_out` metadata to downstream topic.
3. `root_node.py`: same sampling step, then executes the linear query (sum/mean) over the sample using the accumulated weights, calls error estimation, writes `result ± error`.
4. Wire up `kafka_io/producer.py` and `consumer.py` using Faust agents or raw `confluent-kafka-python`, one topic per topology layer as in the paper's Figure 5.
5. Docker Compose: broker + 8 source containers + 4 L1 + 2 L2 + 1 root, with `tc-netem` latency injection matching §V-A (20/40/80ms).

**Acceptance criteria**: a synthetic Gaussian stream flows end-to-end from source containers to root container and produces `SUM ± error` output.

### Phase 4 — SRS baseline
1. Implement coin-flip simple random sampling baseline per node, matching the paper's §IV-B-II description, for head-to-head comparison in all experiments.

### Phase 5 — Reproduce core experiments
Implement and run (on your scaled-down topology — don't worry about matching absolute throughput numbers, only trends):
1. `exp1_sampling_fraction.py` — accuracy loss & throughput vs. sampling fraction (10%–90%), Gaussian + Poisson, ApproxIoT vs SRS vs native. Reproduces Fig. 6–9.
2. `exp2_window_size.py` — latency vs. window size at fixed 10% sampling fraction. Reproduces Fig. 10.
3. `exp3_fluctuating_rates.py` — 3 rate settings (A:B:C:D) as in §V-D, accuracy comparison.
4. `exp4_skew.py` — extreme skew scenario (80/19.89/0.1/0.01% split), reproduce the dramatic accuracy gap vs SRS.
5. `exp5_real_world.py` — NYC taxi substitute dataset + pollution substitute dataset, accuracy/throughput vs sampling fraction.

**Acceptance criteria**: your results show the *same qualitative pattern* as the paper (ApproxIoT accuracy loss consistently lower than SRS, gap widening under skew; throughput comparable across ApproxIoT/SRS/native). Exact numbers will differ due to scale — that's expected and should be stated explicitly in your write-up, not hidden.

### Phase 6 — NOVELTY: Adaptive Budget Controller
This is the new contribution. Goal: remove manual sampling-fraction tuning by closing the loop between the error-estimation module's output and each node's `budget` input.

1. **Design** (`controller/adaptive_budget.py`):
   - Controller runs at the root (or optionally per-node, see "stretch" below) on a fixed cadence (e.g., every K windows).
   - Input: target error bound `ε_target` (user-specified, e.g., "accuracy loss ≤ 0.1%"), current observed error from `error_bounds.py`, current sampling fraction.
   - Output: new sampling fraction/budget for the next interval.
2. **Implement at least two controller policies** in `controller/policies.py` so you can compare them:
   - **Policy A — Proportional feedback (simplest, implement first)**: if `observed_error > ε_target`, increase sampling fraction by a step proportional to the error gap; if well under target, decrease fraction to save resources. Essentially a P-controller on error.
   - **Policy B — PID controller**: add integral and derivative terms for smoother convergence and to avoid oscillation under fluctuating input rates (ties directly into Phase 5's exp3 fluctuating-rate scenario — the controller should adapt as sub-stream rates shift).
   - **(Stretch) Policy C — simple RL/bandit approach**: treat sampling-fraction selection as a multi-armed bandit over a discrete set of fractions, reward = `-( |error - ε_target| + λ * resource_cost )`. Only attempt this after A and B work and are evaluated.
3. **Propagation**: when the controller changes budget at the root, it must propagate the new budget down to edge nodes (extend the control-plane: either a separate lightweight control topic, or piggyback on existing metadata). Document this clearly — it's a real design decision, not a footnote.
4. **Stability concern to handle explicitly**: avoid oscillation/thrashing. Add a minimum dwell time between budget changes and/or a dead-band around the target.

**Acceptance criteria**: under `exp3_fluctuating_rates` conditions (rates shifting over time) and under a step-change injected mid-run, the controller should converge sampling fraction toward a value that keeps observed error near `ε_target`, while a static/manual baseline either over-samples (wasting resources) or under-samples (violating the error target) during the shift.

### Phase 7 — Evaluate the novelty
1. `exp6_adaptive_controller.py`: run the same workloads as Phase 5 but with (a) static manually-tuned budget [[the paper's original approach]] vs (b) your adaptive controller, under:
   - steady-state input (controller should converge and stay near target, minimal overhead vs static)
   - step-change in input rate (controller should re-converge within N windows; static baseline should visibly violate the error target or over-provision)
   - the extreme-skew scenario from exp4 (controller should still hit error target without being told the skew in advance)
2. Metrics to report: time-to-converge, resource savings (bandwidth/CPU) vs a conservatively-set static budget, error-target violation rate, oscillation/stability (variance of sampling fraction over time).
3. Produce comparison plots: error vs. time (controller vs static, overlaid), sampling fraction vs. time, resource usage vs. static baseline.

### Phase 8 — Write-up scaffolding
1. `README.md` should include: system architecture diagram description (text), how to run each experiment, how your scaled-down setup maps back to the original paper's claims, and a clearly labeled "Differences from original paper" section (stack substitutions, scale, dataset substitutions).
2. `notebooks/results_analysis.ipynb`: load all `results/*.csv`, regenerate figures analogous to the paper's Fig. 6–12 plus your new controller figures, side-by-side where possible.

---

## 4. Key Algorithm Reference (for the agent implementing this — exact formulas from the paper)

- Reservoir sampling: keep first `R` items; for item `i > R`, keep with probability `R/i`, replace random existing item.
- Per-substream weight (single node, Eq. 1): `w_i = c_i/N_i` if `c_i > N_i`, else `1`.
- Effective weight: `W_i_out = W_i_in * w_i`.
- Approximate sum (Eq. 2–3): `SUM_i = (Σ_{k=1}^{Y_i} I_i,k) * W_i_out`, total `SUM_* = Σ SUM_i`.
- Multi-node weight calibration under async windows (Eq. 9): `W_i,j_out = W_i,j_in * w_i,j * (C_i,j_in / c_i,j)`.
- Variance of approximate sum (Eq. 11): `Var(SUM) = Σ_i [ c_i,src * (c_i,src - Y_i,j) * s_i,j² / Y_i,j ]`, where `s_i,j²` is sample variance (Eq. 12).
- Variance of approximate mean (Eq. 14): weighted combination using `φ_i = c_i,src / Σc_i,src`.
- Error bound: sqrt(variance) scaled by 1/2/3 for 68%/95%/99.7% confidence (standard three-sigma rule).

---

## 5. Experimental Parameters (match the paper where feasible)

- Synthetic Gaussian sub-streams: A(μ=10,σ=5), B(μ=1000,σ=50), C(μ=10000,σ=500), D(μ=100000,σ=5000).
- Synthetic Poisson sub-streams: A(λ=10), B(λ=100), C(λ=1000), D(λ=10000).
- Skew experiment: A(λ=10, 80% of volume), B(λ=100, 19.89%), C(λ=1000, 0.1%), D(λ=10,000,000, 0.01%).
- Fluctuating-rate settings: Setting1 (50k:25k:12.5k:625), Setting2 (25k:25k:25k:25k), Setting3 (625:12.5k:25k:50k) items/sec.
- Sampling fractions tested: 10%, 20%, ..., 90%.
- Window size default: 1 second (vary 0.5s–5s for exp2).
- Network latency: 20ms (source→L1), 40ms (L1→L2), 80ms (L2→root).
- Error target(s) for controller experiments: try at least two, e.g., 0.1% and 0.5% accuracy loss targets, to show the controller adapts its sampling fraction accordingly.

---

## 6. Success Criteria / Definition of Done

- [ ] All Phase 1–4 unit tests pass.
- [ ] End-to-end pipeline runs in Docker Compose and produces `result ± error` at the root for a synthetic stream.
- [ ] Experiments 1–5 reproduce the paper's *qualitative* trends (documented with your own plots, explicitly noting scale differences from the original).
- [ ] Adaptive controller (Phase 6) converges to target error under both steady-state and step-change conditions, with evidence of resource savings vs. a static baseline.
- [ ] README clearly documents what was replicated, what was substituted, and what is novel.
- [ ] All code has tests; all experiments are reproducible via a single `make experiments` or `python -m experiments.run_all` style entry point.

---

## 7. Suggested Build Order for the Agent

1. Phase 1 (algorithms) → 2 (error estimation) fully unit-tested, no infra.
2. Phase 4 (SRS baseline) — trivial, do alongside Phase 1.
3. Phase 3 (node workflow + Kafka/Redpanda wiring) — get one source → one root working first before scaling to full topology.
4. Scale Phase 3 to full 4-layer topology in Docker Compose.
5. Phase 5 (reproduce experiments) — validate the replication before touching novelty.
6. Phase 6–7 (adaptive controller + its evaluation) — the actual contribution.
7. Phase 8 (write-up scaffolding) last.

Do not start the controller (Phase 6) until Phase 5 experiments show sane, paper-consistent trends — the controller evaluation is only meaningful once the baseline replication is trustworthy.
