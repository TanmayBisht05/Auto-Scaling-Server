"""
Purpose: Scaling actuator. Polls current system metrics, requests a scaling
decision from the brain server (AI mode) or applies a static threshold policy
(static mode), and executes Docker Compose commands to adjust replica count.
 
Usage:
    python autoscaler.py                                              # AI mode, ai_traffic_data.csv
    python autoscaler.py --mode static --data data/static_traffic_data.csv
"""
import argparse
import time
import requests
import subprocess
import os

# --- CONFIGURATION ---
BRAIN_URL = "http://localhost:6000/decide"
MIN_SERVERS = 1
MAX_SERVERS = 10
INTERVAL = 5  # Seconds between decisions

# ─────────────────────────────────────────────────────────────────────────────
#  STATIC THRESHOLD CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────
 
class StaticController:
    CPU_UP    = 70.0
    CPU_DOWN  = 30.0
    LAT_UP    = 1000.0
    LAT_DOWN  = 300.0
    COOLDOWN  = 3
 
    def __init__(self):
        self._cooldown = 0
 
    def decide(self, metrics: dict) -> dict:
        action = "HOLD"
        if self._cooldown > 0:
            self._cooldown -= 1
        else:
            cpu     = metrics['current_cpu']
            latency = metrics['latency']
            if cpu > self.CPU_UP or latency > self.LAT_UP:
                action         = "SCALE_UP"
                self._cooldown = self.COOLDOWN
            elif cpu < self.CPU_DOWN and latency < self.LAT_DOWN:
                action         = "SCALE_DOWN"
                self._cooldown = self.COOLDOWN
        return {"action": action, "fuzzy_score": 0.0,
                "estimated_capacity": 0.0, "predicted_rps": int(metrics['current_rps'])}
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_metrics_from_csv(data_file: str) -> dict | None:
    try:
        if not os.path.exists(data_file):
            return None
        with open(data_file, 'r') as f:
            rows = f.readlines()
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


def get_metrics(data_file: str):
    try:
        container_ids = get_server_container_ids()
        replicas      = len(container_ids)
        if replicas == 0:
            return None
 
        cmd_cpu    = ["docker", "stats", "--no-stream",
                      "--format", "{{.CPUPerc}}"] + container_ids
        output_cpu = subprocess.check_output(cmd_cpu).decode().strip().split("\n")
 
        cpus = []
        for line in output_cpu:
            try:
                cpus.append(float(line.replace("%", "").strip()))
            except ValueError:
                pass
        avg_cpu = sum(cpus) / len(cpus) if cpus else 0.0
 
        csv_m = get_metrics_from_csv(data_file)
        if csv_m:
            current_rps = csv_m['rps']
            fail_ratio  = csv_m['fail_ratio']
            avg_latency = csv_m['latency']
        else:
            current_rps = avg_latency = fail_ratio = 0
 
        return {
            "replicas":    replicas,
            "current_cpu": avg_cpu,
            "current_rps": current_rps,
            "latency":     avg_latency,
            "fail_ratio":  fail_ratio,
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
        
        # Force Nginx to re-resolve the backend IPs
        print("Reloading Nginx configuration...")
        os.system("docker compose exec -T nginx nginx -s reload")
    else:
        print("Scaling limit reached.")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
 
def run_controller(mode: str, data_file: str):
    static_ctrl = StaticController()
 
    print(f"Auto-Scaler started — mode: {mode.upper()}")
    print(f"Reading metrics from: {data_file}")
    if mode == 'ai':
        print(f"Connecting to Brain at: {BRAIN_URL}")
 
    while True:
        try:
            metrics = get_metrics(data_file)
 
            if metrics:
                if mode == 'static':
                    decision = static_ctrl.decide(metrics)
                    action   = decision["action"]
                    print(
                        f"RPS={int(metrics['current_rps'])} "
                        f"CPU={int(metrics['current_cpu'])}% "
                        f"Latency={int(metrics['latency'])}ms | "
                        f"Static: {action}"
                    )
                    if action != "HOLD":
                        scale_docker(metrics['replicas'], action)
 
                else:
                    try:
                        resp = requests.post(BRAIN_URL, json=metrics)
                        if resp.status_code == 200:
                            decision = resp.json()
                            action   = decision.get("action", "HOLD")
                            print(
                                f"RPS={int(metrics['current_rps'])} "
                                f"CPU={int(metrics['current_cpu'])}% "
                                f"Latency={int(metrics['latency'])}ms | "
                                f"Brain: {action} "
                                f"(Score:{decision.get('fuzzy_score',0)}) "
                                f"Est.Cap:{decision.get('estimated_capacity',0)}"
                            )
                            if action != "HOLD":
                                scale_docker(metrics['replicas'], action)
                        else:
                            print(f"Brain error {resp.status_code}: {resp.text}")
                    except requests.exceptions.ConnectionError:
                        print("Cannot reach Brain Server on port 6000.")
            else:
                print("Waiting for metrics...")
 
        except KeyboardInterrupt:
            print("\nStopping Auto-Scaler.")
            break
        except Exception as e:
            print(f"Controller error: {e}")
 
        time.sleep(INTERVAL)
 
 
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
 
    parser = argparse.ArgumentParser(description="Auto-Scaler actuator")
    parser.add_argument(
        '--mode', choices=['ai', 'static'], default='ai',
        help='ai = Brain Server (default);  static = threshold policy'
    )
    parser.add_argument(
        '--data',
        default=os.path.join(SCRIPT_DIR, "data/ai_traffic_data.csv"),
        help='CSV file written by collector.py (default: ai_traffic_data.csv)'
    )
    args = parser.parse_args()
    run_controller(mode=args.mode, data_file=os.path.abspath(args.data))
