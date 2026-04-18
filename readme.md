# Intelligent Auto-Scaling System (Hybrid Soft Computing Approach)

## Purpose
This project implements a proactive, adaptive, and self-tuning horizontal autoscaler for microservice architectures. Traditional autoscalers are reactive; they scale infrastructure only after a load spike has occurred, resulting in latency penalties during container initialization. 

This system solves the initialization delay by employing a hybrid soft computing pipeline:
1. **Neural Networks (NN):** Predicts future traffic loads (Requests Per Second) based on historical telemetry windows.
2. **Fuzzy Logic (FL):** Evaluates predicted traffic and real-time CPU utilization to make nuanced scaling decisions (SCALE_UP, SCALE_DOWN, HOLD) without thrashing.
3. **Genetic Algorithms (GA):** Continuously optimizes the six core parameters of the Fuzzy Inference System (load thresholds and CPU boundaries) to minimize SLA violations and resource waste.

## Architecture & File Structure

The repository is divided into three primary operational domains.

### 1. Infrastructure & Target Application
* `backend/index.js`: Node.js application simulating a microservice workload. It combines asynchronous I/O wait times with synchronous CPU-bound mathematical calculations to accurately mimic real-world container stress.
* `nginx/nginx.conf`: Reverse proxy load balancer distributing incoming traffic across active backend replicas.
* `Dockerfile` & `docker-compose.yml`: Orchestration configurations defining the scalable `server` service and the static `nginx` entrypoint.

### 2. Telemetry & Load Generation
* `locust/traffic_profile.py`: Pre-processes the WorldCup98 dataset logs into a normalized, second-by-second RPS profile.
* `locust/locustfile_worldcup.py`: Locust load generator. It utilizes a concurrency pool of 100 synchronous users and dynamic wait-time calculations to replay the time-compressed historical traffic profile against the infrastructure.
* `locust/collector.py`: Telemetry agent. Parses Nginx access logs to calculate RPS and latency, and polls the Docker daemon for container CPU utilization. Outputs state to `traffic_data.csv`.

### 3. AI Controller & Soft Computing
* `ai-controller/brain_server.py`: REST API inference engine. Integrates the NN predictor and FL decision-maker. Provides scaling commands to the actuator.
* `ai-controller/autoscaler.py`: The control loop actuator. Polls real-time metrics, requests decisions from the brain server, and executes `docker compose scale` commands.
* `ai-controller/train_brain.py`: Offline optimization script to train the PyTorch predictor on historical time-series data.
* `ai-controller/optimizer.py`: Genetic Algorithm module. Evolves the FL membership function parameters to minimize a cost function of latency penalties and server waste.
* `ai-controller/benchmark.py`: Comparative evaluation tool to analyze Hybrid AI performance against a static threshold baseline.
* `ai-controller/modules/predictor.py`: PyTorch Feedforward Neural Network definition.
* `ai-controller/modules/fuzzy_logic.py`: Fuzzy Inference System defining antecedents, consequents, and rule blocks.

## Execution Pipeline

Execute these phases in order from the project root directory to initialize the environment, train the models, and run the real-time simulation.

### Phase 1: Infrastructure Verification
Rebuild and start the containers. Verify the backend is responding and benchmark baseline performance.

```bash
docker compose down
docker compose build
docker compose up -d --scale server=2

# Verify response
curl http://localhost:8080/api

# Quick concurrent load test (ensure 0 failures)
ab -n 200 -c 20 http://localhost:8080/api
```

### Phase 2: Generate Traffic Profiles
Process the raw WorldCup dataset into training and testing CSV profiles.

```bash
cd locust
python traffic_profile.py --csv ../ai-controller/data/worldcup/invocation_count.csv --out traffic_profile_train.csv --split train --max-rps 400
python traffic_profile.py --csv ../ai-controller/data/worldcup/invocation_count.csv --out traffic_profile_test.csv --split test --max-rps 400
cd ..
```

### Phase 3: Train Neural Network Predictor
Train the PyTorch model on the generated training profile.

```bash
cd ai-controller
python train_brain.py --profile ../locust/traffic_profile_train.csv
cd ..
```

### Phase 4: Clean Stale Telemetry
Ensure the autoscaler and collector start with a clean state.

```bash
rm ai-controller/data/traffic_data.csv
```

### Phase 5: Start the Control Loop (5 Terminals)
Open 5 separate terminal windows. Execute these commands sequentially.

`Terminal 1 (Docker)`: Reset to a single replica baseline.

```bash
docker compose down
docker compose build
docker compose up -d --scale server=1
```

`Terminal 2 (Brain Server)`: Start the inference engine.

```bash
cd ai-controller
source ../venv/bin/activate
python brain_server.py
```

`Terminal 3 (Autoscaler)`: Start the actuator.

```bash
cd ai-controller
source ../venv/bin/activate
python autoscaler.py
```

`Terminal 4 (Telemetry Collector)`: Start logging metrics.

```bash
cd locust
source ../venv/bin/activate
python collector.py
```

`Terminal 5 (Load Generator)`: Execute the time-compressed test profile. Wait for initialization logs to confirm simulation acceleration.

```bash
cd locust
source ../venv/bin/activate
TEST_DURATION_SEC=3600 python -m locust -f locustfile_worldcup.py --headless -u 100 -r 100 --host http://localhost:8080 --run-time 60m
```

### Phase 6: Benchmarking
To properly benchmark the proactive AI against a reactive static model, you must run the Phase 5 loop twice (once with the AI, once enforcing a static rule in autoscaler.py) and save the resulting traffic_data.csv files independently before running benchmark.py.