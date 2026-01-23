import requests
import subprocess
import time
from datetime import datetime

LOCUST_URL = "http://localhost:8089/stats/requests"

def get_locust_stats():
    r = requests.get(LOCUST_URL).json()

    stats = r["stats"]
    if stats:
        avg_latency = sum(s["avg_response_time"] for s in stats) / len(stats)
    else:
        avg_latency = 0.0

    return {
        "rps": r["total_rps"],
        "latency": avg_latency
    }





def get_cpu_usage():
    cmd = [
        "docker", "stats",
        "--no-stream",
        "--format", "{{.Name}},{{.CPUPerc}}"
    ]
    output = subprocess.check_output(cmd).decode().strip().split("\n")

    cpus = []
    for line in output:
        if "autoscaling-server-server" in line:
            _, cpu = line.split(",")
            cpus.append(float(cpu.replace("%", "")))

    return sum(cpus) / len(cpus) if cpus else 0.0




def get_replica_count():
    cmd = [
        "docker", "ps",
        "--filter", "name=autoscaling-server-server",
        "--format", "{{.Names}}"
    ]
    output = subprocess.check_output(cmd).decode().strip()
    return len(output.split("\n")) if output else 0






if __name__ == "__main__":
    start_time = time.time()

    print("time_offset,rps,cpu,replicas,latency")

    while True:
        locust = get_locust_stats()
        cpu = get_cpu_usage()
        replicas = get_replica_count()
        t = round(time.time() - start_time, 2)

        row = f"{t},{locust['rps']},{cpu},{replicas},{locust['latency']}"
        print(row)

        with open("metrics.csv", "a") as f:
            f.write(row + "\n")

        time.sleep(1)
