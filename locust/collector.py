import requests
import subprocess
import time
import os
import sys

# --- SMART PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "../ai-controller/data/traffic_data.csv")
DATA_FILE = os.path.abspath(DATA_FILE)

LOCUST_URL = "http://localhost:8089/stats/requests"

print(f"DEBUG: Saving data to: {DATA_FILE}")


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


def get_locust_stats():
    try:
        r = requests.get(LOCUST_URL).json()
        current_rps = r.get("total_rps", 0)
        fail_ratio = r.get("fail_ratio", 0)

        stats = r.get("stats", [])
        if stats:
            total_reqs = sum(s["num_requests"] for s in stats)
            if total_reqs > 0:
                avg_latency = sum(s["avg_response_time"] * s["num_requests"]
                                  for s in stats) / total_reqs
            else:
                avg_latency = 0.0
        else:
            avg_latency = 0.0

        return {
            "rps": current_rps,
            "latency": avg_latency,
            "fail_ratio": fail_ratio
        }
    except Exception:
        return {"rps": 0, "latency": 0, "fail_ratio": 0}


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


if __name__ == "__main__":
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

    print(f"Collector started. Writing to {DATA_FILE}")
    print("Waiting for Locust...")

    start_time = time.time()

    while True:
        try:
            stats = get_locust_stats()
            cpu = get_cpu_usage()
            replicas = get_replica_count()

            t = round(time.time() - start_time, 2)

            row = f"{t},{stats['rps']},{cpu},{replicas},{stats['latency']},{stats['fail_ratio']}"

            print(row)

            with open(DATA_FILE, "a") as f:
                f.write(row + "\n")
                f.flush()
                os.fsync(f.fileno())

            time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping collector...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
