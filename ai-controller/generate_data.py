import pandas as pd
import numpy as np
import random
import os

# Configuration
OUTPUT_FOLDER = "data"
OUTPUT_FILE = "traffic_data.csv"
TOTAL_MINUTES = 60       # Generate 1 hour of data
INTERVAL_SECONDS = 5     # Data point every 5 seconds


def generate_traffic():
    # Ensure the data folder exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder: {OUTPUT_FOLDER}")

    # Calculate total steps
    total_steps = (TOTAL_MINUTES * 60) // INTERVAL_SECONDS

    # 1. Time Vector
    time_indices = np.arange(total_steps)

    # 2. Base Traffic (Sine Wave for "Day/Night" cycle)
    # We simulate a "peak" happening in the middle of our hour
    base_load = 500 + 300 * np.sin(time_indices / 100)

    data = []

    # Start with 5 servers
    current_replicas = 5

    for t, load in zip(time_indices, base_load):
        # --- A. Generate Random Traffic ---
        noise = random.randint(-50, 50)

        # 10% chance of a "Micro Spike" (user click burst)
        micro_spike = 150 if random.random() > 0.90 else 0

        # 1% chance of "Huge Spike" (Black Friday event)
        mega_spike = 800 if random.random() > 0.99 else 0

        rps = max(50, int(load + noise + micro_spike + mega_spike))

        # --- B. Simulate System Physics ---
        # Each server can handle ~150 RPS comfortably.
        # If RPS > Capacity, CPU goes up.

        server_capacity = 150
        total_capacity = current_replicas * server_capacity

        # Utilization Ratio (e.g., 1.2 means we are 20% over capacity)
        utilization = rps / total_capacity

        # CPU Usage Model
        # Base CPU is 20% (idle overhead) + 60% * utilization
        cpu = 20 + (60 * utilization)

        # Add random CPU jitter (+/- 5%)
        cpu += random.uniform(-5, 5)

        # Cap CPU at 100% (Physics limit)
        cpu = np.clip(cpu, 10, 100)

        # --- C. Simulate Latency (The Critical Metric) ---
        # Latency stays low (40ms) until CPU hits ~80%, then it explodes exponentially.
        base_latency = 40  # ms

        if cpu < 80:
            # Healthy state: linear small increase
            latency = base_latency + (cpu / 10)
        else:
            # Overload state: Exponential barrier
            # (cpu - 80) can be up to 20. 20^1.8 is roughly 220.
            # Latency will spike to ~300ms+ at 100% CPU.
            latency = base_latency + 10 + ((cpu - 80) ** 1.8)

        # Add random network jitter
        latency += random.uniform(-5, 5)

        data.append([t * INTERVAL_SECONDS, rps, round(cpu, 2),
                    current_replicas, round(latency, 2)])

    # --- D. Save to CSV ---
    df = pd.DataFrame(
        data, columns=["time_offset", "rps", "cpu", "replicas", "latency"])

    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
    df.to_csv(output_path, index=False)
    print(f"Success! Generated {len(df)} rows of training data.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    generate_traffic()
