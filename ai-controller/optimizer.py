"""
Purpose: Genetic Algorithm optimization routine. Evolves a 6-parameter vector
for the Fuzzy Inference System to minimize a cost function incorporating
latency penalties and resource waste. Supports offline execution and an online
daemon thread for continuous adaptation.
Usage:
    python optimizer.py                          # uses ai_traffic_data.csv by default
    python optimizer.py --data data/custom.csv
"""

import argparse
import json
import os
import threading
import time
import warnings

import numpy as np
import pandas as pd
import pygad
import requests

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "data/ai_traffic_data.csv")
PARAM_SAVE_PATH  = os.path.join(SCRIPT_DIR, "models/fuzzy_params.json")
BRAIN_RELOAD_URL = "http://localhost:6000/reload_params"

# ── Cost weights ──────────────────────────────────────────────────────────────
W_LATENCY = 1.0      # cost per ms of simulated latency
W_SLA     = 1000.0   # heavy penalty when simulated latency > 2000 ms
W_SERVER  = 20.0     # cost per active server per step

# ── Online GA settings ────────────────────────────────────────────────────────
ONLINE_INTERVAL = 120   # seconds between adaptation cycles
ONLINE_WINDOW   = 200   # most-recent rows to optimise on
_online_running = False

# ── Gene bounds (pygad gene_space format) ─────────────────────────────────────
#   0: load_low      1: load_high
#   2: cpu_safe      3: cpu_danger
#   4: thresh_up     5: |thresh_down|  (stored positive, negated when used)
GENE_SPACE = [
    {'low': 0.20, 'high': 0.90},   # load_low
    {'low': 0.50, 'high': 1.80},   # load_high
    {'low': 10.0, 'high': 60.0},   # cpu_safe
    {'low': 50.0, 'high': 95.0},   # cpu_danger
    {'low': 0.20, 'high': 0.80},   # thresh_up   (positive)
    {'low': 0.10, 'high': 0.80},   # |thresh_down| (pygad needs positive ranges;
                                   #  we negate it when passing to FuzzyBrain)
]


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_resample(data_path: str, n_rows: int = 999_999) -> pd.DataFrame:
    if not os.path.exists(data_path):
        return pd.DataFrame()

    df = pd.read_csv(
        data_path,
        header=None,
        names=['time', 'rps', 'cpu', 'replicas', 'latency', 'fail_ratio'],
        on_bad_lines='skip'   # drop any corrupted lines like line 15
    )

    df = df.tail(n_rows).copy()

    # time is elapsed seconds since collector start, not a unix timestamp.
    # Convert to a relative datetime so resample() works correctly.
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime')

    df = df[['rps', 'cpu', 'replicas', 'latency', 'fail_ratio']].resample('5s').mean()
    df = df.interpolate(method='linear').dropna()
    return df.reset_index(drop=True)


def load_recent_rows(n_rows: int = ONLINE_WINDOW,
                     data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    return _load_and_resample(data_path, n_rows=n_rows)

# ─────────────────────────────────────────────────────────────────────────────
#  CAPACITY ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_capacity(df: pd.DataFrame) -> float:
    """
    Derive a realistic RPS-per-replica capacity from observed healthy periods.

    Logic:
      1. Filter rows where the system was clearly healthy:
           latency  < 500 ms   (well under SLA)
           fail_ratio < 0.005  (< 0.5 % errors)
           replicas >= 1       (sanity)
      2. Compute load_per_replica = rps / replicas for each healthy row.
      3. Return the 95th percentile of those values.
         - Using the 95th (not max) avoids outlier spikes skewing the estimate.
         - It represents "the load each replica handled comfortably almost all
           the time", which is what the GA simulation should use.

    Fallback: if there aren't enough healthy rows (e.g. data was all high-load),
    return 10.0 so the simulation stays runnable rather than crashing.
    """
    LATENCY_HEALTHY   = 500.0
    FAIL_RATIO_HEALTHY = 0.005
    MIN_HEALTHY_ROWS  = 10     # need at least this many rows to trust the estimate

    healthy = df[
        (df['latency']    < LATENCY_HEALTHY)  &
        (df['fail_ratio'] < FAIL_RATIO_HEALTHY) &
        (df['replicas']   >= 1)
    ].copy()

    if len(healthy) < MIN_HEALTHY_ROWS:
        print(f"[GA] Warning: only {len(healthy)} healthy rows — using fallback 50.0 rps/replica.")
        return 50.0

    healthy['load_per_replica'] = healthy['rps'] / healthy['replicas'].clip(lower=1)
    est = float(np.percentile(healthy['load_per_replica'], 95))

    # Sanity bounds: a single container handling < 1 or > 10 000 rps/s would be
    # a data error, not a real measurement.
    est = float(np.clip(est, 1.0, 10_000.0))

    print(f"[GA] Empirical capacity estimate: {est:.2f} rps/replica "
          f"(from {len(healthy)} healthy rows, p95)")
    return est


# ─────────────────────────────────────────────────────────────────────────────
#  FITNESS FUNCTION  (uses _estimate_capacity instead of any hardcoded number)
# ─────────────────────────────────────────────────────────────────────────────

def _make_fitness_func(df: pd.DataFrame):
    """
    Factory that closes over `df`.
    est_cap is computed once from the data and shared across all GA evaluations
    in this cycle — it does not change between individuals.
    """
    sim_data = df.iloc[:500].copy()

    # Derive capacity from data once — not hardcoded
    est_cap = _estimate_capacity(df)

    def fitness_func(ga_instance, solution, solution_idx):
        from modules.fuzzy_logic import FuzzyBrain

        l_low, l_high, c_safe, c_danger, t_up, t_down_abs = solution

        if l_low >= l_high:    return 1e-9
        if c_safe >= c_danger: return 1e-9
        if l_low < 0.05:       return 1e-9

        thresh_up   =  abs(float(t_up))
        thresh_down = -abs(float(t_down_abs))

        brain = FuzzyBrain([l_low, l_high, c_safe, c_danger, thresh_up, thresh_down])

        total_cost       = 0.0
        current_replicas = 5

        for _, row in sim_data.iterrows():
            rps = float(row.get('rps', 0))
            cpu = float(row.get('cpu', 0))

            load_ratio = rps / max(current_replicas * est_cap, 0.1)
            score      = brain.compute(load_ratio, cpu)

            if score > thresh_up and current_replicas < 10:
                current_replicas += 1
            elif score < thresh_down and current_replicas > 1:
                current_replicas -= 1

            util        = rps / max(current_replicas * est_cap, 0.1)
            sim_latency = 40 + (util ** 2) * 50 if util <= 1.0 else 40 + util * 200

            cost_lat  = sim_latency * W_LATENCY
            cost_sla  = W_SLA if sim_latency > 2000 else 0
            cost_serv = current_replicas * W_SERVER

            total_cost += cost_lat + cost_sla + cost_serv

        return 1.0 / (total_cost + 1e-8)

    return fitness_func


# ─────────────────────────────────────────────────────────────────────────────
#  PERSIST PARAMS
# ─────────────────────────────────────────────────────────────────────────────

def save_params(solution):
    """Write all 6 genes to fuzzy_params.json."""
    l_low, l_high, c_safe, c_danger, t_up, t_down_abs = solution
    params = {
        "load_low":    round(float(l_low),          4),
        "load_high":   round(float(l_high),         4),
        "cpu_safe":    round(float(c_safe),         4),
        "cpu_danger":  round(float(c_danger),       4),
        "thresh_up":   round(abs(float(t_up)),      4),
        "thresh_down": round(-abs(float(t_down_abs)), 4),
    }
    os.makedirs(os.path.dirname(PARAM_SAVE_PATH), exist_ok=True)
    with open(PARAM_SAVE_PATH, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"[GA] Params saved  →  {PARAM_SAVE_PATH}")
    print(f"     {params}")
    return params


def _notify_brain():
    """Ask the running brain_server to hot-reload fuzzy params."""
    try:
        r = requests.post(BRAIN_RELOAD_URL, timeout=2)
        if r.status_code == 200:
            print("[GA] Brain hot-reloaded successfully.")
        else:
            print(f"[GA] Brain reload returned HTTP {r.status_code}.")
    except Exception as e:
        print(f"[GA] Could not reach brain server: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  CORE GA RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_optimization(df: pd.DataFrame,
                     num_generations: int = 20,
                     sol_per_pop:     int = 10,
                     verbose:         bool = True):
    """Run pygad on `df` and return the best solution array (6 genes)."""
    if df.empty:
        print("[GA] No data — skipping.")
        return None

    fitness_func = _make_fitness_func(df)

    def on_generation(ga):
        if verbose and ga.generations_completed % 5 == 0:
            best_fit = ga.best_solution()[1]
            print(f"  [GA] Gen {ga.generations_completed:3d}  "
                  f"Fitness: {best_fit:.8f}")

    ga_instance = pygad.GA(
        num_generations    = num_generations,
        num_parents_mating = max(2, sol_per_pop // 3),
        fitness_func       = fitness_func,
        sol_per_pop        = sol_per_pop,
        num_genes          = 6,           # ← was 4
        gene_space         = GENE_SPACE,
        mutation_num_genes = 2,           # mutate 2 genes per step (was 1)
        on_generation      = on_generation if verbose else None,
        suppress_warnings  = True,
    )

    ga_instance.run()

    solution, fitness, _ = ga_instance.best_solution()
    if verbose:
        print(f"\n[GA] Evolution complete.  Best fitness: {fitness:.8f}")
        print(f"[GA] Best genes: {[round(float(g), 4) for g in solution]}")

    return solution


# ─────────────────────────────────────────────────────────────────────────────
#  ONLINE CONTINUOUS ADAPTATION  (Objective A)
# ─────────────────────────────────────────────────────────────────────────────

def run_online_ga(interval:  int = ONLINE_INTERVAL,
                  window:    int = ONLINE_WINDOW,
                  data_path: str = DEFAULT_DATA_PATH):
    """
    Daemon thread that re-evolves fuzzy params every `interval` seconds
    and hot-reloads brain_server. Called once from brain_server.py at startup.
    """
    global _online_running
    if _online_running:
        print("[GA] Online loop already active — ignoring duplicate call.")
        return
    _online_running = True
 
    def _loop():
        print(f"[GA] Online adaptation thread started "
              f"(every {interval}s, window={window}, data={data_path})")
        while True:
            time.sleep(interval)
            try:
                print("[GA] Starting online optimisation cycle...")
                df = load_recent_rows(window, data_path=data_path)
 
                if len(df) < 20:
                    print(f"[GA] Only {len(df)} rows available — skipping.")
                    continue
 
                solution = run_optimization(df, num_generations=15,
                                            sol_per_pop=8, verbose=False)
                if solution is not None:
                    save_params(solution)
                    _notify_brain()
                    print("[GA] Online cycle complete.")
 
            except Exception as e:
                print(f"[GA] Online cycle error: {e}")
 
    threading.Thread(target=_loop, daemon=True, name="online-ga").start()


# ─────────────────────────────────────────────────────────────────────────────
#  OFFLINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline GA optimisation")
    parser.add_argument(
        '--data', default=DEFAULT_DATA_PATH,
        help='CSV file to optimise on (default: ai_traffic_data.csv)'
    )
    args = parser.parse_args()
 
    print("=== Offline GA Optimisation ===")
    print(f"Loading data from {args.data} ...")
    df = _load_and_resample(args.data)
    print(f"  {len(df)} rows after resampling.")
 
    solution = run_optimization(df, num_generations=20, sol_per_pop=10, verbose=True)
 
    if solution is not None:
        save_params(solution)
        print("\nDone. Restart brain_server.py to apply the new parameters.")
    else:
        print("Optimisation failed — no data available.")