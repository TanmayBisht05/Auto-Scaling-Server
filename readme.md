# Intelligent Auto-Scaling System (Hybrid Soft Computing Approach)

## Purpose

Traditional autoscalers are reactive — they scale infrastructure only *after* a load spike has occurred, incurring latency penalties during container initialization. This project implements a proactive, adaptive, and self-tuning horizontal autoscaler for microservice architectures that acts *before* saturation occurs.

The system employs a hybrid soft computing pipeline:

1. **Neural Network (NN):** Predicts future traffic (RPS) from a sliding window of historical telemetry, giving the system a look-ahead signal.
2. **Fuzzy Logic (FL):** Evaluates predicted load and real-time CPU utilisation to make nuanced scaling decisions (SCALE_UP / SCALE_DOWN / HOLD) without thrashing.
3. **Genetic Algorithm (GA):** Continuously evolves the six core FL membership-function parameters to minimise SLA violations and resource waste, adapting to observed traffic patterns every two minutes.

### Infrastructure & Target Application

- **`backend/main.go`** — Go HTTP server replacing the original Node.js backend. Uses `runtime.GOMAXPROCS(runtime.NumCPU())` so goroutines spread across all host cores, making CPU a real, correlated signal for the fuzzy system. Each request combines async I/O sleep with CPU-bound math work.
- **`nginx/nginx.conf`** — Reverse proxy using Docker's internal DNS resolver (`127.0.0.11`) so nginx re-resolves `server:5000` per request and picks up newly scaled containers within seconds.
- **`Dockerfile`** — Multi-stage Go build. Produces a minimal Alpine runtime image.
- **`docker-compose.yml`** — Defines the scalable `server` service with `cpus: '2.0'` (enforced by plain compose, unlike `deploy.resources`) and the static `nginx` entrypoint.

### Telemetry & Load Generation

- **`locust/traffic_profile.py`** — Converts the WorldCup98 `invocation_count.csv` into a `(second_offset, rps)` profile CSV. Supports train/test splits and RPS scaling via `--max-rps`.
- **`locust/locustfile_worldcup.py`** — Locust load generator. Reads `NUM_USERS` and `TEST_DURATION_SEC` from environment variables (both must be set). Per-user wait time is derived as `NUM_USERS / target_rps` so the profile's RPS is the total across all users, not per-user. Clock starts on the first actual request, not at module load, to avoid profile offset during ramp-up.
- **`locust/collector.py`** — Telemetry agent. Reads Nginx access logs for RPS and fail ratio; probes `http://localhost:8080/api` for latency; polls `docker stats` for CPU. Writes one row per second to a configurable output CSV via `--output`.

### AI Controller & Soft Computing

- **`ai-controller/brain_server.py`** — Flask REST API. Loads the NN predictor and GA-optimised fuzzy params. On each `/decide` request: corrects RPS for failures, predicts next-tick demand, computes fuzzy score, applies panic overrides. Dynamically learns per-server capacity via asymmetric EWMA (only updates upward when healthy load exceeds current estimate; only updates downward when overloaded below current estimate). Accepts `--data` to point its online GA at the right CSV.
- **`ai-controller/autoscaler.py`** — Control loop. Runs every 5 seconds. `--mode ai` posts metrics to the brain server; `--mode static` applies a local CPU/latency threshold policy (mirrors the `StaticController` used in benchmarking). Reads metrics from the CSV specified by `--data`.
- **`ai-controller/train_brain.py`** — Trains the PyTorch load predictor. `--profile` for initial training from a WorldCup profile CSV; `--data` for retraining on a live collector CSV.
- **`ai-controller/optimizer.py`** — Genetic Algorithm over 6 fuzzy parameters. Run offline via CLI or continuously as a daemon thread inside the brain server. Uses empirical capacity estimation (p95 of healthy `rps/replica` rows) rather than hardcoded values.
- **`ai-controller/benchmark.py`** — Compares two real CSV files (one per controller run) using actual recorded latency, fail_ratio, and replica counts. Capacity is estimated empirically from the AI run's healthy periods and is used only for the resource-waste metric.
- **`ai-controller/modules/predictor.py`** — Three-layer feedforward NN: input window → 32 → 16 → 1.
- **`ai-controller/modules/fuzzy_logic.py`** — Fuzzy Inference System. Antecedents: load ratio, CPU. Consequent: scaling action score in [-1, +1]. Four rules covering scale-up, scale-down (two variants), and hold. GA optimises the six boundary parameters including the up/down thresholds.

## Execution Pipeline

Run all commands from the project root unless stated otherwise.

### Phase 1 — Capacity Benchmark (one-time, determines max-rps)

Build the Go image and measure single-container throughput before setting traffic profile parameters.

```bash
docker compose build --no-cache
docker compose up -d --scale server=1
sleep 5
curl http://localhost:8080/api   # sanity check

# Run each line and note "Requests per second" and "Time per request"
ab -n 500  -c 20  http://localhost:8080/api 2>&1 | grep -E "Requests per second|Time per request|Failed"
ab -n 1000 -c 50  http://localhost:8080/api 2>&1 | grep -E "Requests per second|Time per request|Failed"
ab -n 2000 -c 100 http://localhost:8080/api 2>&1 | grep -E "Requests per second|Time per request|Failed"
ab -n 2000 -c 150 http://localhost:8080/api 2>&1 | grep -E "Requests per second|Time per request|Failed"
ab -n 2000 -c 200 http://localhost:8080/api 2>&1 | grep -E "Requests per second|Time per request|Failed"
```

While each `ab` run is active, check CPU in a second terminal:

```bash
docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

**Reading the results:** Find the concurrency level where RPS plateaus and latency starts rising sharply — that is your single-container capacity (`CAP`). Then compute:

```bash
CAP=600          # replace with your measured value
MAX_REPLICAS=10
echo "Recommended --max-rps: $(echo "$CAP * $MAX_REPLICAS * 0.75" | bc)"
# → use this value in Phase 2

AVG_LATENCY_MS=110   # replace with your measured mean latency under load
PEAK_RPS=$(echo "$CAP * $MAX_REPLICAS * 0.75" | bc)
echo "Recommended NUM_USERS / -u: $(echo "$PEAK_RPS / (1000 / $AVG_LATENCY_MS)" | bc)"
# → use this value in Phase 5 and 6
```

```bash
docker compose down
```

---

### Phase 2 — Generate Traffic Profile

```bash
cd locust

python traffic_profile.py \
    --csv   ../ai-controller/data/worldcup/invocation_count.csv \
    --out   traffic_profile_train.csv \
    --split train \
    --max-rps 3000        # replace with value from Phase 1

python traffic_profile.py \
    --csv   ../ai-controller/data/worldcup/invocation_count.csv \
    --out   traffic_profile_test.csv \
    --split test \
    --max-rps 3000        # replace with value from Phase 1

# Confirm output
head -3 traffic_profile_train.csv
wc -l  traffic_profile_train.csv
head -3 traffic_profile_test.csv
wc -l  traffic_profile_test.csv

cd ..
```

---

### Phase 3 — Train Neural Network

```bash
cd ai-controller
source ../venv/bin/activate

python train_brain.py --profile ../locust/traffic_profile_train.csv

# Expected output: loss decreasing across 100 epochs, final loss < 0.01
ls -lh models/load_predictor.pth

cd ..
```

---

### Phase 4 — Seed Fuzzy Parameters (offline GA)

Run once before the first live test to give the brain server a starting point better than defaults.

```bash
cd ai-controller
source ../venv/bin/activate

python optimizer.py   # reads ai_traffic_data.csv if it exists, otherwise uses defaults

cat models/fuzzy_params.json

cd ..
```

---

### Phase 5 — AI Controller Run (4 terminals)

Clean state first:

```bash
rm -f ai-controller/data/ai_traffic_data.csv
docker compose down
docker compose up -d --scale server=1
sleep 5
curl http://localhost:8080/api
```

Open 4 terminals and run one command per terminal in order. Wait for each to be ready before starting the next.

**Terminal 1 — Brain server:**
```bash
cd ai-controller
source ../venv/bin/activate
python brain_server.py --data data/ai_traffic_data.csv
```
Wait for: `Neural Network Loaded` and `Online adaptation thread started`.

**Terminal 2 — AI autoscaler:**
```bash
cd ai-controller
source ../venv/bin/activate
python autoscaler.py --mode ai --data data/ai_traffic_data.csv
```

**Terminal 3 — Telemetry collector:**
```bash
cd locust
source ../venv/bin/activate
python collector.py --output ../ai-controller/data/ai_traffic_data.csv
```
Wait for rows to appear (one per second) before starting load.

**Terminal 4 — Load generator:**
```bash
cd locust
source ../venv/bin/activate
NUM_USERS=500 TEST_DURATION_SEC=3600 \
python -m locust -f locustfile_worldcup.py \
    --headless -u 500 -r 500 \
    --host http://localhost:8080 \
    --run-time 60m
```

Replace `NUM_USERS=500` and `-u 500 -r 500` with the value calculated in Phase 1. They must match.

After the 60-minute Locust run completes:

```bash
echo "AI run rows:" && wc -l ai-controller/data/ai_traffic_data.csv
docker compose down
```

Stop terminals 1, 2, 3 with `Ctrl+C`.

---

### Phase 6 — Static Controller Run (3 terminals)

No brain server needed.

```bash
rm -f ai-controller/data/static_traffic_data.csv
docker compose down
docker compose up -d --scale server=1
sleep 5
curl http://localhost:8080/api
```

**Terminal 1 — Static autoscaler:**
```bash
cd ai-controller
source ../venv/bin/activate
python autoscaler.py --mode static --data data/static_traffic_data.csv
```

**Terminal 2 — Telemetry collector:**
```bash
cd locust
source ../venv/bin/activate
python collector.py --output ../ai-controller/data/static_traffic_data.csv
```

**Terminal 3 — Load generator (identical profile and duration):**
```bash
cd locust
source ../venv/bin/activate
NUM_USERS=500 TEST_DURATION_SEC=3600 \
python -m locust -f locustfile_worldcup.py \
    --headless -u 500 -r 500 \
    --host http://localhost:8080 \
    --run-time 60m
```

After completion:

```bash
echo "Static run rows:" && wc -l ai-controller/data/static_traffic_data.csv
docker compose down
```

---

### Phase 7 — Benchmark Comparison

```bash
cd ai-controller
source ../venv/bin/activate

mkdir -p ../results

python benchmark.py \
    --ai-data     data/ai_traffic_data.csv \
    --static-data data/static_traffic_data.csv \
    --out         ../results/benchmark_report.txt

cat ../results/benchmark_report.txt

cd ..
```

---

## Benchmark Results

Results from a 60-minute run using WorldCup98 knockout-round traffic (July 1–26 1998, scaled to 3000 RPS peak), replayed against a 2-CPU Go backend with up to 10 replicas. 552 ticks evaluated per controller.

```
==================================================================
  BENCHMARK: Hybrid AI  vs  Static Threshold Scaler
  Metrics are from real recorded values — no simulation.
==================================================================

  SLA Violations (%)
    Hybrid AI :     0.000  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    Static    :     0.180  ██████████████████████████████
    → Better: Hybrid AI

  Mean Resource Waste
    Hybrid AI :     0.002  █████████░░░░░░░░░░░░░░░░░░░░░
    Static    :     0.007  ██████████████████████████████
    → Better: Hybrid AI

  Latency Mean (ms)
    Hybrid AI :    46.300  █████████████████████████████░
    Static    :    47.300  ██████████████████████████████
    → Better: Hybrid AI

  Latency Std-Dev (ms)
    Hybrid AI :     5.800  ████████████████░░░░░░░░░░░░░░
    Static    :    10.800  ██████████████████████████████
    → Better: Hybrid AI

  Mean Fail Ratio
    Hybrid AI :     0.000  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    Static    :     0.000  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    → Better: Tie

  Mean Replicas Used
    Hybrid AI :     1.140  █████████████████████████████░
    Static    :     1.180  ██████████████████████████████
    → Better: Hybrid AI

==================================================================
  Ticks evaluated : AI=552  Static=552
  Hybrid AI won   : 5/6 metrics
==================================================================
```

**Reading the results:**

- **SLA Violations (0.000% vs 0.180%):** The AI controller recorded zero SLA breaches. The static controller's reactive nature caused it to lag behind traffic spikes, allowing latency to exceed 2000ms on 0.18% of ticks before scale-up triggered.
- **Latency Std-Dev (5.8ms vs 10.8ms):** The AI's proactive scaling produces a flatter latency profile — nearly half the variance of the static controller. This is the most direct evidence of the predictive advantage: containers are ready before queues form.
- **Latency Mean (46.3ms vs 47.3ms):** A modest 1ms difference in mean, reflecting that the static controller recovered reasonably well between spikes — the advantage of proactive scaling is primarily in eliminating the spike itself, not in steady-state performance.
- **Resource Waste:** The AI used fractionally fewer excess replicas (0.002 vs 0.007 mean waste), demonstrating that proactive scaling does not require over-provisioning to avoid SLA violations.
- **Fail Ratio (Tie):** Both controllers achieved zero mean fail ratio — the Go backend handled queue buildup without errors in both cases, so this metric did not differentiate them.
- **Mean Replicas (1.140 vs 1.180):** The static controller kept slightly more replicas running on average despite worse SLA performance, consistent with the pattern of scaling up reactively and being slow to scale back down.

---

## Static Controller Policy (for reference)

The baseline uses a simple cooldown-gated threshold policy, identical in both `autoscaler.py --mode static` and `benchmark.py`:

| Condition | Action | Cooldown |
|-----------|--------|----------|
| CPU > 70% **or** Latency > 1000ms | SCALE_UP | 3 ticks |
| CPU < 30% **and** Latency < 300ms | SCALE_DOWN | 3 ticks |
| Otherwise | HOLD | — |

---

## Retraining on Live Data

After completing Phase 5, the AI controller can be retrained on real observed traffic for subsequent runs:

```bash
cd ai-controller
source ../venv/bin/activate

# Retrain NN on real AI-run data
python train_brain.py --data data/ai_traffic_data.csv

# Re-run offline GA on real data to update fuzzy params
python optimizer.py --data data/ai_traffic_data.csv

cd ..
```

Then restart the brain server for the new model to take effect.