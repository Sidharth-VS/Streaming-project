# Running and testing the ApproxIoT replication

Commands for the in-process replication, the experiment suite, the results
notebook and the containerized (Kafka/Redpanda) deployment, plus what to look
out for and the current status of the novelty module.

Requires Python ≥ 3.10. Docker is needed only for the deployment path.

## 1. Setup

```bash
cd Streaming-project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,realworld]"

# Only needed to run a node on the host (Docker installs it inside the image):
pip install -e ".[kafka]"
```

## 2. Tests

```bash
python -m pytest
# or
make test
```

Expected: **63 passed** in about 1 second.

Targeted subsets:

```bash
python -m pytest tests/test_reservoir.py tests/test_weighted_hierarchical.py
python -m pytest tests/test_error_bounds.py tests/test_srs.py
python -m pytest tests/test_adaptive_controller.py tests/test_runtime.py
python -m pytest tests/test_latency.py tests/test_simulation.py
python -m pytest tests/test_kafka_transport.py
```

## 3. Experiment suite

```bash
python -m experiments.run_all
# or
make experiments
```

Expected wall time **~3.5 minutes** (EXP1 alone is ~2.5 min because it runs at
the paper's ~11k items/s operating point). Expected output:

```text
[    ok] exp1_sampling_fraction          145s
[    ok] exp2_window_size                 34s
[    ok] exp3_fluctuating_rates            2s
[    ok] exp4_skew                         1s
[    ok] exp5_real_world                  15s
[    ok] exp6_adaptive_controller          8s
Summary written to .../results/summary.json
```

Individual experiments:

```bash
python -m experiments.exp1_sampling_fraction
python -m experiments.exp2_window_size
python -m experiments.exp3_fluctuating_rates
python -m experiments.exp4_skew
python -m experiments.exp5_real_world
python -m experiments.exp6_adaptive_controller
```

Different seed or shorter run:

```bash
python -m experiments.run_all --seed 7 --windows 32
```

## 4. Inspect results

```bash
ls results/
cat results/summary.json

# Verdict table — every experiment should print pass=True
python - <<'PY'
import json
s = json.load(open("results/summary.json"))
for name, payload in s.items():
    q = payload.get("qualitative_check") or {}
    print(f"{name:30s} pass={q.get('pass')}")
PY
```

Artifacts:

| File | What it shows |
|---|---|
| `results/exp1_loss_vs_fraction_{gaussian,poisson}.png` | accuracy loss vs sampling fraction (Fig. 6) |
| `results/exp1_throughput_vs_fraction_{gaussian,poisson}.png` | throughput vs fraction (Fig. 7), ~11k items/s |
| `results/exp2_window_size.png` | measured latency vs window size (Fig. 10) |
| `results/exp3_fluctuating_rates.png` | Gaussian + Poisson rate-setting panels (Fig. 11a/b) |
| `results/exp4_skew.png` | extreme-skew gap (Fig. 11c analogue) |
| `results/exp5_real_world.png` | real-world workload (Fig. 12) |
| `results/exp6_adaptive_controller.png` | controller error / sampling fraction over time |
| `results/summary.json` | all verdicts plus per-experiment metrics |

## 5. Notebook

```bash
jupyter notebook notebooks/results_analysis.ipynb
# or
jupyter lab notebooks/results_analysis.ipynb
```

Install Jupyter if needed (`pip install jupyter`) and run the experiment suite
first so the CSVs exist.

## 6. Containerized deployment (Redpanda + 15 nodes)

Short smoke run (20 windows × 2 s ≈ 40 s of simulated time):

```bash
MAX_WINDOWS=20 docker compose up -d --build
# or
make docker-smoke
```

Full run (60 windows):

```bash
docker compose up -d --build
# or
make docker-up
```

Monitor and collect:

```bash
docker compose ps
docker compose logs -f root                  # Ctrl-C stops following only
docker compose logs --tail=20 source-a1 l1-1 l2-1

docker compose down                          # stop the broker when done
```

### Extracting / recording the results

Two artifacts are produced by the root node, nothing else has to be collected:

| Artifact | Written by | Contents |
|---|---|---|
| `results/docker_results.csv` | root container (`RESULTS_PATH`) | one row per window: estimate, error bound, relative stderr, sampled count, fraction, bandwidth saved, e2e latency |
| root log line (JSON) | root container, on exit | totals + medians for the whole run |

The extractor turns both into a single recordable file, `results/docker_summary.json`:

```bash
python scripts/summarize_docker_results.py      # or: make docker-results
```

It skips the warm-up windows (`n_in_root == 0`), prints the per-window table, then
aggregates accuracy vs the 0.5 % target, bandwidth saved, latency and controller
activity, and ends with `pass: true/false`:

```json
{
  "windows_with_results": 55,
  "median_relative_stderr": 0.004088,
  "windows_within_target_pct": 72.7,
  "median_latency_ms": 4350.6,
  "final_fraction": 0.675,
  "bandwidth_saved_total": 0.341611,
  "checks": {"data_flowed": true, "accuracy_target_met": true,
             "bandwidth_saved_positive": true, "controller_acted": true},
  "pass": true
}
```

Other ways to get the same numbers:

```bash
make docker-csv                                   # raw per-window CSV
docker compose logs root | tail -1                # the JSON summary line
docker compose logs root | grep '^{' > results/docker_root_log.json
```

`results/docker_results.csv` is **overwritten on every run**, so copy
`results/docker_summary.json` somewhere else if you need to keep several runs.

Expected startup line in every node container:

```text
[entrypoint] netem: eth0 egress delay 20ms   # 40ms for L1, 80ms for L2
```

Expected final root summary:

```json
{"node":"root","windows":20,"windows_with_results":17,
 "median_relative_stderr":0.0066,"controller_num_changes":3,
 "final_fraction":0.675,"median_latency_ms":3048.5}
```

The 15 node containers exit when their window count is reached; **Redpanda keeps
running**, which is why a foreground `docker compose up` appears to hang. Use
`-d` and stop it with `docker compose down`.

## 7. What to look out for

- **EXP1 runtime:** it deliberately runs at ~11k items/s, so ~2.5 minutes is
  normal, not a hang.
- **Warm-up windows:** the first ~3 root windows have `n_in_root = 0`; only
  compare rows with `n_in_root > 0`.
- **EXP2 latency:** ApproxIoT grows roughly linearly (~1.7 s → ~19 s from
  0.5 s to 5 s windows); SRS must stay flat at ~140 ms. If SRS grows, the
  windowless SRS branch of `src/approxiot/latency.py` was changed.
- **Throughput:** reported as simulated items/second and should stay flat at
  ~11k. Python wall-clock speed is a separate diagnostic
  (`wall_clock_throughput_items_per_sec`) and must not be compared to the paper.
- **EXP5 datasets:** `proxy_used` in the JSON says whether real TLC/OpenAQ data
  was loaded. Use `download=True` with `pyarrow` installed for real data;
  otherwise documented distribution-matched proxies are used.
- **EXP6 tight targets:** the 0.1% target is infeasible at this scale, so
  `adaptive-*-tight` saturates at fraction 1.0 and can show high violation
  rates. Use the 0.5% regimes for the convergence claim;
  `time_to_reconverge_windows` is in the JSON.
- **`env: 'bash\r': No such file or directory`:** the entrypoint was checked
  out with CRLF endings (Windows `core.autocrlf=true`), so every node exits
  127 before Python starts. `.gitattributes` forces `*.sh` to LF and the
  Dockerfile strips stray `\r` while building, so a rebuild fixes it; if you
  edit the script on Windows, re-run `git add --renormalize scripts/`.
- **Docker build:** if `--build` is slow it is downloading numpy /
  confluent-kafka. The deployment needs a working Docker daemon and
  `NET_ADMIN` for `tc-netem`; without it containers warn and continue without
  emulated latency. **Port 9092** must be free and Redpanda wants ~1 GB.
- **`results/docker_results.csv`** is overwritten on every Docker run and is
  owned by root inside the container.
- **Cleanup:** `make clean` removes generated experiment outputs;
  `docker compose down` removes the stack. Neither touches source files.

## 8. Novelty: closed-loop adaptive budget controller

Implemented in both the simulation and the Kafka deployment path.

| Piece | File |
|---|---|
| Controller (trailing median filter, dead-band, dwell time, confirmation-before-actuation, clamps, history/oscillation) | `src/approxiot/controller/adaptive_budget.py` |
| Policies: A proportional, B PID in log-fraction space with anti-windup, C ε-greedy bandit (stretch) | `src/approxiot/controller/policies.py` |
| Online error signal `sqrt(Var)/|result|` | `src/approxiot/error_estimation/error_bounds.py` |
| Control plane (`ControlMessage` on a dedicated `control` topic) | `src/approxiot/kafka_io/transport.py`, `simulation.py` (`_apply_control`), `runtime/node.py` (`apply_control`) |
| Deployment wiring | `docker-compose.yml` root env: `TARGET_ERROR`, `POLICY`, `CADENCE_WINDOWS`, `MIN_DWELL_WINDOWS`, `DEAD_BAND` |
| Evaluation | `experiments/exp6_adaptive_controller.py` |

Evidence:

- `python -m experiments.exp6_adaptive_controller` compares static-conservative
  / static-tuned against adaptive-P / adaptive-PID at **0.5% and 0.1% targets**
  over steady state, a mid-run rate step-change and extreme skew, reporting
  bandwidth saved, violation rate, fraction stability, median error and
  time-to-reconverge.
- The EXP6 verdict in `results/summary.json` passes.
- `tests/test_adaptive_controller.py` + `tests/test_runtime.py` (15 tests)
  cover policies, convergence, dwell time, dead-band, clamps, control-message
  emission and root-side broadcasting.
- The live Docker run showed the controller acting: `controller_num_changes` > 0,
  fraction 0.5 → 0.37 → 0.675, median relative stderr ≈0.004 against the 0.005
  target.

Known gaps (by design, not defects):

- Bandit policy C is implemented and unit-tested but not part of the EXP6
  comparison (the plan marks it a stretch goal).
- The controller runs system-wide at the root and broadcasts one fraction; a
  per-node controller is not implemented.
- The 0.1% target is infeasible at this replication scale; tight regimes
  saturate at fraction 1.0.
- The live runtime feeds the controller per-window stderr, while the simulation
  feeds a trailing-window aggregate; both converge, values differ slightly.

## 9. Verification log

| Check | Command | Result |
|---|---|---|
| Unit/integration tests | `python -m pytest` | 63 passed (exit 0) |
| Experiment suite | `python -m experiments.run_all` | exit 0, all six verdicts pass |
| Notebook cells | executed against `results/` | all code cells run |
| Compose file | `docker compose config --quiet` | valid |
| Makefile targets | `make -n docker-up docker-smoke docker-logs docker-results docker-down` | correct commands |
| Entrypoint syntax | `bash -n scripts/docker-entrypoint.sh` | OK |
| Live stack | `MAX_WINDOWS=20 docker compose up -d --build` | 15 nodes + Redpanda ran; controller changed budgets; latency recorded per window |

Known limitations:

- A **from-scratch `docker build`** may be slow or fail on a throttled network
  (the numpy wheel). The validated run used an incrementally built image on the
  same base; `docker compose config` and the full stack were verified.
- The **latency trace does not model CPU queue build-up** on a saturated
  cluster, so the 10%-sampling speed-up over native is ~1.2–1.4× rather than
  the paper's 6×. The window-size ordering and flat SRS behaviour are
  reproduced and checked by the EXP2 verdict.
- **Real NYC TLC/OpenAQ downloads** were not exercised; EXP5 uses the
  documented proxy fallback unless `download=True` is passed with `pyarrow`.
- **EXP3–6** run at a reduced `SIM_RATE_SCALE` for runtime, with relative rates
  and distributions preserved; EXP1/EXP2 run at the paper's ~11k items/s.
