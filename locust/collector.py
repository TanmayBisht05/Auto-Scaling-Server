"""
Purpose: Telemetry agent. Parses Nginx access logs to calculate RPS and failure
ratios. Polls the Docker daemon for CPU utilization across active backend
replicas. Measures active application latency and writes aggregated state to a
CSV file.
Usage:
    python collector.py                                    # writes ai_traffic_data.csv
    python collector.py --output ../ai-controller/data/static_traffic_data.csv
"""

import argparse
import collections
import requests
import subprocess
import time
import os

# --- CONFIGURATION ---
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
NGINX_CONTAINER = "auto-scaling-server-nginx-1"
RPS_WINDOW_SECONDS = 5



def get_server_container_ids():
    """
    Finds all container IDs belonging to the 'server' service 
    using the standard Docker Compose label.
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


def get_nginx_logs(since_seconds: int = RPS_WINDOW_SECONDS) -> list[str]:
    """Fetch recent Nginx access log lines via docker logs --since."""
    try:
        cmd = [
            "docker", "logs",
            "--since", f"{since_seconds}s",
            NGINX_CONTAINER
        ]
        output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT
        ).decode(errors="replace")
        return [l for l in output.strip().split("\n") if l.strip()]
    except Exception:
        return []


def get_locust_stats() -> dict:
    """
    Measure RPS and latency directly from Nginx access logs.
    No dependency on Locust's web API.
    """
    lines = get_nginx_logs(since_seconds=RPS_WINDOW_SECONDS)

    # Each line looks like:
    # 172.19.0.1 - - [11/Apr/2026:21:42:25 +0000] "GET /api HTTP/1.1" 200 137 ...
    rps         = len(lines) / RPS_WINDOW_SECONDS
    fail_count  = 0
    total_count = len(lines)

    for line in lines:
        try:
            # Status code is the 7th space-separated token
            parts = line.split()
            status = int(parts[8])
            if status >= 500:
                fail_count += 1
        except (IndexError, ValueError):
            continue

    fail_ratio = fail_count / total_count if total_count > 0 else 0.0

    # Latency: Nginx doesn't log it by default, so we measure it with
    # a direct probe request to the backend
    latency = 0.0
    try:
        t0 = time.time()
        requests.get("http://localhost:8080/api", timeout=2)
        latency = (time.time() - t0) * 1000   # convert to ms
    except Exception:
        latency = 0.0

    return {
        "rps":        round(rps, 2),
        "latency":    round(latency, 2),
        "fail_ratio": round(fail_ratio, 4),
    }


def get_cpu_usage():
    ids = get_server_container_ids()
    if not ids:
        return 0.0

    try:
        # Get CPU stats for ONLY the identified containers
        # We pass the list of IDs directly to docker stats
        cmd = ["docker", "stats", "--no-stream",
               "--format", "{{.CPUPerc}}"] + ids
        output = subprocess.check_output(cmd).decode().strip().split("\n")

        cpus = []
        for line in output:
            try:
                # Remove % and convert to float
                val = float(line.replace("%", "").strip())
                cpus.append(val)
            except ValueError:
                pass

        return sum(cpus) / len(cpus) if cpus else 0.0
    except Exception:
        return 0.0


def get_replica_count():
    return len(get_server_container_ids())


def run(output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Collector started. Writing to {output_path}")
 
    start_time = time.time()
 
    while True:
        try:
            stats    = get_locust_stats()
            cpu      = get_cpu_usage()
            replicas = get_replica_count()
            t        = round(time.time() - start_time, 2)
 
            row = f"{t},{stats['rps']},{cpu},{replicas},{stats['latency']},{stats['fail_ratio']}"
            print(row)
 
            with open(output_path, "a") as f:
                f.write(row + "\n")
                f.flush()
                os.fsync(f.fileno())
 
            time.sleep(1)
 
        except KeyboardInterrupt:
            print("\nStopping collector.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telemetry collector")
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "../ai-controller/data/ai_traffic_data.csv"),
        help="Path to write CSV output (default: ai_traffic_data.csv)",
    )
    args = parser.parse_args()
    run(output_path=os.path.abspath(args.output))
