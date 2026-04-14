"""
Purpose: Scaling actuator. Polls current system metrics from Docker and Locust, requests a scaling decision from the brain server API, and executes Docker Compose commands to adjust the active replica count.
Usage: Run continuously to close the control loop between telemetry, inference, and infrastructure scaling.
"""

import csv
import time
import requests
import subprocess
import os
import sys

# --- CONFIGURATION ---
BRAIN_URL = "http://localhost:6000/decide"
LOCUST_URL = "http://localhost:8089/stats/requests"
MIN_SERVERS = 1
MAX_SERVERS = 10
INTERVAL = 5  # Seconds between decisions
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data/traffic_data.csv")

def get_metrics_from_csv() -> dict | None:
    """Read the most recent row from traffic_data.csv as fallback."""
    try:
        if not os.path.exists(DATA_FILE):
            return None
        with open(DATA_FILE, 'r') as f:
            rows = f.readlines()
        if not rows:
            return None
        # Get last non-empty line
        last = next((r.strip() for r in reversed(rows) if r.strip()), None)
        if not last:
            return None
        parts = last.split(',')
        if len(parts) < 6:
            return None
        return {
            'rps':        float(parts[1]),
            'cpu':        float(parts[2]),
            'replicas':   float(parts[3]),
            'latency':    float(parts[4]),
            'fail_ratio': float(parts[5]),
        }
    except Exception:
        return None


def get_server_container_ids():
    """
    Finds container IDs for the 'server' service using Docker Compose labels.
    Fixes the issue where container names mismatch (e.g. auto-scaling vs autoscaling).
    """
    try:
        cmd = [
            "docker", "ps", "-q",
            "--filter", "label=com.docker.compose.service=server"
        ]
        output = subprocess.check_output(cmd).decode().strip()
        if not output:
            return []
        return output.split("\n")
    except Exception:
        return []


def get_metrics():
    """Gather all metrics needed by the brain"""
    try:
        # 1. Get Replica Count & Container IDs
        container_ids = get_server_container_ids()
        replicas = len(container_ids)
        if replicas == 0:
            return None  # System is down or starting up

        # 2. Get CPU Usage (Avg of all replicas)
        # We pass the list of IDs directly to docker stats to be precise
        cmd_cpu = ["docker", "stats", "--no-stream",
                   "--format", "{{.CPUPerc}}"] + container_ids
        output_cpu = subprocess.check_output(
            cmd_cpu).decode().strip().split("\n")

        cpus = []
        for line in output_cpu:
            try:
                val = float(line.replace("%", "").strip())
                cpus.append(val)
            except ValueError:
                pass
        avg_cpu = sum(cpus) / len(cpus) if cpus else 0.0

        csv_metrics = get_metrics_from_csv()
        if csv_metrics:
            current_rps = csv_metrics['rps']
            fail_ratio  = csv_metrics['fail_ratio']
            avg_latency = csv_metrics['latency']
        else:
            try:
                r = requests.get(LOCUST_URL, timeout=1).json()
                current_rps = r.get("total_rps", None) or r.get("current_rps", 0)
                fail_ratio  = r.get("fail_ratio", 0)
                stats = r.get("stats", [])
                avg_latency = 0.0
                if stats:
                    total = sum(s["num_requests"] for s in stats)
                    if total > 0:
                        avg_latency = sum(
                            s["avg_response_time"] * s["num_requests"]
                            for s in stats) / total
            except requests.exceptions.RequestException:
                current_rps = 0
                fail_ratio  = 0
                avg_latency = 0

        return {
            "replicas": replicas,
            "current_cpu": avg_cpu,
            "current_rps": current_rps,
            "latency": avg_latency,
            "fail_ratio": fail_ratio
        }
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None


def scale_docker(current_replicas, action):
    target = current_replicas
    if action == "SCALE_UP":
        target += 1
    elif action == "SCALE_DOWN":
        target -= 1

    target = max(MIN_SERVERS, min(target, MAX_SERVERS))

    if target != current_replicas:
        print(f"Executing: Scaling from {current_replicas} to {target} servers...")
        os.system(f"docker compose up -d --scale server={target}")
    else:
        print("Scaling limit reached (Min/Max).")


def run_controller():
    print("Auto-Scaler Controller Started... Monitoring system.")
    print(f"Connecting to Brain at: {BRAIN_URL}")

    while True:
        try:
            metrics = get_metrics()

            if metrics:
                # Ask the Brain for a decision
                try:
                    resp = requests.post(BRAIN_URL, json=metrics)

                    if resp.status_code == 200:
                        decision = resp.json()
                        action = decision.get("action", "HOLD")
                        score = decision.get("fuzzy_score", 0)
                        est_cap = decision.get("estimated_capacity", 0)
                        pred_rps = decision.get("predicted_rps", 0)

                        # Print Status Line
                        print(f"Stats: RPS={int(metrics['current_rps'])} CPU={int(metrics['current_cpu'])}% Latency={int(metrics['latency'])}ms | "
                              f"Brain: {action} (Score: {score}) | Est.Cap: {est_cap}")

                        if action != "HOLD":
                            scale_docker(metrics['replicas'], action)
                    else:
                        print(f"Brain Error {resp.status_code}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    print(
                        "Could not connect to Brain Server. Is it running on port 6000?")
            else:
                print("Waiting for Docker/Locust metrics...")

        except KeyboardInterrupt:
            print("\nStopping Auto-Scaler.")
            break
        except Exception as e:
            print(f"Unexpected Controller Error: {e}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    run_controller()
