"""
benchmark.py  —  Comparative Analysis
──────────────────────────────────────
Compares two real traffic CSV files produced by running the system under
identical load — once with the AI controller, once with the static controller.
 
Latency, fail_ratio, and replica counts are taken directly from the recorded
CSV values. This is correct because each CSV reflects what actually happened
during that run — the AI's lower latency is real evidence of proactive scaling,
not an artefact to be normalised away.
 
Capacity per server is estimated empirically from the AI run's healthy periods
and is used only for the resource-waste calculation.
 
Usage:
    python benchmark.py \
        --ai-data     data/ai_traffic_data.csv \
        --static-data data/static_traffic_data.csv \
        --out         ../results/benchmark_report.txt
"""

import argparse
import csv
import math
import os
import statistics

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline='') as f:
        sample = f.read(256)
        f.seek(0)
        has_header = not sample.split('\n')[0].split(',')[0].replace('.','').isdigit()

        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        'rps':        float(row.get('rps', 0)),
                        'cpu':        float(row.get('cpu', 0)),
                        'replicas':   float(row.get('replicas', 1)),
                        'latency':    float(row.get('latency', 0)),
                        'fail_ratio': float(row.get('fail_ratio', 0)),
                    })
                except (ValueError, KeyError):
                    continue
        else:
            reader = csv.reader(f)
            for row in reader:
                try:
                    if len(row) < 6:
                        continue
                    rows.append({
                        'rps':        float(row[1]),
                        'cpu':        float(row[2]),
                        'replicas':   float(row[3]),
                        'latency':    float(row[4]),
                        'fail_ratio': float(row[5]),
                    })
                except (ValueError, IndexError):
                    continue
    return rows

# ─────────────────────────────────────────────────────────────────────────────
#  CAPACITY ESTIMATION  (used only for resource-waste calculation)
# ─────────────────────────────────────────────────────────────────────────────
 
def estimate_capacity(rows: list[dict]) -> float:
    """
    Derive RPS-per-replica from healthy periods in the AI run.
    Healthy = latency < 500 ms AND fail_ratio < 0.5 %.
    Returns the p95 of load_per_replica during those periods.
    Only used to judge how many replicas were "needed" per tick for
    the resource-waste metric — not used for latency or SLA calculations.
    """
    healthy_loads = [
        r['rps'] / max(r['replicas'], 1)
        for r in rows
        if r['latency'] < 500 and r['fail_ratio'] < 0.005 and r['replicas'] >= 1
    ]
    if len(healthy_loads) < 10:
        print(f"[Benchmark] Warning: only {len(healthy_loads)} healthy rows "
              f"for capacity estimate — using fallback of 50 rps/replica.")
        return 50.0
    cap = float(np.clip(np.percentile(healthy_loads, 95), 1.0, 10_000.0))
    print(f"[Benchmark] Empirical capacity: {cap:.2f} rps/replica "
          f"(p95 of {len(healthy_loads)} healthy rows)")
    return cap
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  METRICS  (all derived from real recorded values)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict], estimated_cap: float) -> dict:
    latencies    = [r['latency']    for r in rows]
    fail_ratios  = [r['fail_ratio'] for r in rows]
    replicas_series = [r['replicas'] for r in rows]
    n = len(rows)

    sla_violations = sum(1 for l in latencies if l > 200)

    waste_per_tick = []
    for row in rows:
        needed = max(1, math.ceil(row['rps'] / max(estimated_cap, 1)))
        waste  = max(0, row['replicas'] - needed - 1)
        waste_per_tick.append(waste)

    return {
        'n_ticks':           n,
        'sla_violation_pct': round(100 * sla_violations / max(n, 1), 2),
        'mean_waste':        round(statistics.mean(waste_per_tick), 3),
        'latency_mean_ms':   round(statistics.mean(latencies), 1),
        'latency_stddev_ms': round(statistics.pstdev(latencies), 1),
        'mean_fail_ratio':   round(statistics.mean(fail_ratios), 5),
        'mean_replicas':     round(statistics.mean(replicas_series), 2),
    }
# ─────────────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────────────
 
def _bar(value: float, max_val: float, width: int = 30) -> str:
    filled = int(round(value / max(max_val, 1e-9) * width))
    return '█' * filled + '░' * (width - filled)
 
 
def print_report(ai: dict, static: dict, out_path: str | None = None):
    lines = [
        "=" * 66,
        "  BENCHMARK: Hybrid AI  vs  Static Threshold Scaler",
        "  Metrics are from real recorded values — no simulation.",
        "=" * 66,
    ]
 
    metrics_to_show = [
        ("SLA Violations (%)",   'sla_violation_pct', True),
        ("Mean Resource Waste",  'mean_waste',        True),
        ("Latency Mean (ms)",    'latency_mean_ms',   True),
        ("Latency Std-Dev (ms)", 'latency_stddev_ms', True),
        ("Mean Fail Ratio",      'mean_fail_ratio',   True),
        ("Mean Replicas Used",   'mean_replicas',     True),
    ]
 
    ai_wins = 0
    for label, key, lower_better in metrics_to_show:
        ai_val     = ai[key]
        static_val = static[key]
        max_val    = max(ai_val, static_val, 1e-9)
 
        lines.append(f"\n  {label}")
        lines.append(f"    Hybrid AI : {ai_val:>9.3f}  {_bar(ai_val, max_val)}")
        lines.append(f"    Static    : {static_val:>9.3f}  {_bar(static_val, max_val)}")
 
        if lower_better:
            winner = ("Hybrid AI" if ai_val < static_val else
                      "Static"   if static_val < ai_val else "Tie")
        else:
            winner = ("Hybrid AI" if ai_val > static_val else
                      "Static"   if static_val > ai_val else "Tie")
 
        if winner == "Hybrid AI":
            ai_wins += 1
        lines.append(f"    → Better: {winner}")
 
    lines += [
        "\n" + "=" * 66,
        f"  Ticks evaluated : AI={ai['n_ticks']}  Static={static['n_ticks']}",
        f"  Hybrid AI won   : {ai_wins}/{len(metrics_to_show)} metrics",
        "=" * 66,
    ]
 
    report = "\n".join(lines)
    print(report)
 
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved → {out_path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare AI vs Static scaler from two real traffic CSV files")
    parser.add_argument(
        '--ai-data', required=True,
        help='CSV from the AI controller run'
    )
    parser.add_argument(
        '--static-data', required=True,
        help='CSV from the static controller run'
    )
    parser.add_argument(
        '--out', default=None,
        help='Optional path to write the text report'
    )
    args = parser.parse_args()
 
    print(f"Loading AI data     : {args.ai_data}")
    ai_rows = load_csv(args.ai_data)
    if not ai_rows:
        print("ERROR: No AI data loaded — check path.")
        return
 
    print(f"Loading Static data : {args.static_data}")
    static_rows = load_csv(args.static_data)
    if not static_rows:
        print("ERROR: No static data loaded — check path.")
        return
 
    # Trim both to the same length for a fair tick-by-tick comparison
    n = min(len(ai_rows), len(static_rows))
    ai_rows     = ai_rows[:n]
    static_rows = static_rows[:n]
    print(f"\n{n} ticks used from each run.\n")
 
    # Capacity estimated from AI run's healthy periods — used only for waste calc
    cap = estimate_capacity(ai_rows)
 
    ai_metrics     = compute_metrics(ai_rows,     cap)
    static_metrics = compute_metrics(static_rows, cap)
 
    print_report(ai_metrics, static_metrics, out_path=args.out)
 
 
if __name__ == '__main__':
    main()
 