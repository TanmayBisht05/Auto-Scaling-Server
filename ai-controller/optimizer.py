from modules.fuzzy_logic import FuzzyBrain
import os
import json
import numpy as np
import pandas as pd
import pygad
import warnings
warnings.filterwarnings("ignore")

# Configuration
DATA_PATH = "data/traffic_data.csv"
PARAM_SAVE_PATH = "models/fuzzy_params.json"

# Weights (The "Personality" of your scaler)
W_LATENCY = 1.0     # Cost per ms of latency
W_SLA = 1000.0      # Huge penalty if latency > 200ms
W_SERVER = 20.0     # Cost per server per step

# Load Data Once
if not os.path.exists(DATA_PATH):
    print("Data not found!")
    exit()
df = pd.read_csv(DATA_PATH, header=None, names=[
                 'time', 'rps', 'cpu', 'replicas', 'latency'])

# 2. Fix the Time (Resample 3.5s -> 5.0s)
# Convert the 'time' column (seconds since start) to a real datetime index
df['datetime'] = pd.to_datetime(df['time'], unit='s', origin='unix')
df = df.set_index('datetime')

# 3. Resample to strict 5-Second Buckets
# This averages out the rows to match your Node.js loop time
df = df.resample('5s').mean()
df = df.interpolate()  # Fill gaps

# 4. Clean up
df = df.reset_index(drop=True)
df = df.dropna()


def fitness_func(ga_instance, solution, solution_idx):
    # 1. Constraints (Don't let genes be invalid)
    l_low, l_high, c_safe, c_danger = solution
    if l_low >= l_high:
        return 1e-6
    if c_safe >= c_danger:
        return 1e-6
    if l_low < 0.1 or l_high > 1.9:
        return 1e-6

    # 2. Setup the Brain with these genes
    brain = FuzzyBrain(solution)

    total_cost = 0.0
    current_replicas = 5  # Start simulation with 5 servers

    # 3. Run the Simulation (Fast Loop)
    # We only run on a subset of data to speed up the GA (first 500 rows)
    sim_data = df.iloc[:500]

    for _, row in sim_data.iterrows():
        # A. Calculate State
        capacity = current_replicas * 150  # 150 reqs/server
        load_ratio = row['rps'] / capacity

        # B. Ask Brain
        decision = brain.compute(load_ratio, row['cpu'])

        # C. Apply Decision
        if decision > 0.6:
            current_replicas += 1
        elif decision < -0.6 and current_replicas > 1:
            current_replicas -= 1

        # D. Calculate COST
        # Latency Estimation (Physics Model)
        util = row['rps'] / (current_replicas * 150)
        sim_latency = 40 + (util**2 * 50) if util > 1 else 40

        cost_lat = sim_latency * W_LATENCY
        cost_sla = W_SLA if sim_latency > 200 else 0
        cost_serv = current_replicas * W_SERVER

        total_cost += (cost_lat + cost_sla + cost_serv)

    # 4. Return Fitness (Inverse of Cost)
    return 1.0 / (total_cost + 1e-8)


def run_optimization():
    print("Starting Genetic Evolution...")

    # Define Gene Ranges
    # Gene 0: Load Low (0.2 - 0.7)
    # Gene 1: Load High (0.6 - 1.5)
    # Gene 2: CPU Safe (20 - 60)
    # Gene 3: CPU Danger (60 - 95)
    gene_space = [
        {'low': 0.2, 'high': 0.7},
        {'low': 0.6, 'high': 1.5},
        {'low': 20, 'high': 60},
        {'low': 60, 'high': 95}
    ]

    ga_instance = pygad.GA(
        num_generations=20,
        num_parents_mating=4,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=4,
        gene_space=gene_space,
        mutation_num_genes=1
    )

    ga_instance.run()

    # Best Solution
    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"Evolution Complete!")
    print(f"Best Genes: {solution}")

    # Save to JSON
    params = {
        "load_low": solution[0], "load_high": solution[1],
        "cpu_safe": solution[2], "cpu_danger": solution[3]
    }
    with open(PARAM_SAVE_PATH, 'w') as f:
        json.dump(params, f)
    print(f"Saved optimized parameters to {PARAM_SAVE_PATH}")


if __name__ == "__main__":
    run_optimization()
