"""
benchmark.py  —  Objective C: Comparative Analysis
────────────────────────────────────────────────────
Replays traffic_data.csv through two controllers:
  A) Hybrid AI   — the real brain_server decisions recorded in the CSV
  B) Static      — a simple threshold scaler (CPU > 70 % → scale up, < 30 % → down)

Computes and prints:
  • SLA violation rate   (% of ticks with latency > 2000 ms)
  • Resource waste score (mean excess replicas above minimum needed)
  • Latency stability    (std-dev of response time in ms)
  • Fail ratio           (mean across all ticks)

Run:
    python benchmark.py [--data ai-controller/data/traffic_data.csv]
                        [--out  results/benchmark_report.txt]
"""

import argparse
import csv
import json
import math
import os
import statistics
from copy import deepcopy


# ─────────────────────────────────────────────────────────────────────────────
#  STATIC REACTIVE THRESHOLD SCALER
# ─────────────────────────────────────────────────────────────────────────────

class StaticScaler:
    """
    Dumb reactive policy:
      scale-up   if CPU > CPU_UP   or  latency > LAT_UP
      scale-down if CPU < CPU_DOWN and latency < LAT_DOWN
      else hold
    Cooldown prevents thrashing (no action within COOLDOWN ticks of last change).
    """
    CPU_UP    = 70.0
    CPU_DOWN  = 30.0
    LAT_UP    = 1000.0   # ms
    LAT_DOWN  = 300.0    # ms
    COOLDOWN  = 3        # ticks

    def __init__(self):
        self.replicas    = 1
        self._cooldown_remaining = 0

    def decide(self, cpu: float, latency: float) -> str:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return "HOLD"

        if cpu > self.CPU_UP or latency > self.LAT_UP:
            self._cooldown_remaining = self.COOLDOWN
            return "SCALE_UP"
        elif cpu < self.CPU_DOWN and latency < self.LAT_DOWN:
            self._cooldown_remaining = self.COOLDOWN
            return "SCALE_DOWN"
        return "HOLD"

    def apply(self, action: str, min_r=1, max_r=10):
        if action == "SCALE_UP"   and self.replicas < max_r:
            self.replicas += 1
        elif action == "SCALE_DOWN" and self.replicas > min_r:
            self.replicas -= 1


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline='') as f:
        # Detect whether file has a header row
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
                        'ai_action':  row.get('action', 'HOLD'),
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
                        'ai_action':  'HOLD',
                    })
                except (ValueError, IndexError):
                    continue
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict], replicas_series: list[int],
                    estimated_cap: float = 10.0) -> dict:
    """
    Given the observed data rows and the replica counts our controller *would
    have chosen*, compute benchmark metrics.
    """
    latencies    = [r['latency']    for r in rows]
    fail_ratios  = [r['fail_ratio'] for r in rows]
    n = len(rows)

    # SLA violations: latency > 2000 ms
    sla_violations = sum(1 for l in latencies if l > 2000)

    # Resource waste: excess replicas above minimum needed per tick
    waste_per_tick = []
    for i, row in enumerate(rows):
        needed = max(1, math.ceil(row['rps'] / max(estimated_cap, 1)))
        waste  = max(0, replicas_series[i] - needed - 1)  # 1 spare OK
        waste_per_tick.append(waste)

    return {
        'n_ticks':           n,
        'sla_violation_pct': round(100 * sla_violations / max(n, 1), 2),
        'mean_waste':        round(statistics.mean(waste_per_tick), 3),
        'latency_mean_ms':   round(statistics.mean(latencies), 1),
        'latency_stddev_ms': round(statistics.pstdev(latencies), 1),
        'mean_fail_ratio':   round(statistics.mean(fail_ratios), 5),
        'max_replicas':      max(replicas_series),
        'mean_replicas':     round(statistics.mean(replicas_series), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATE STATIC SCALER ON DATA
# ─────────────────────────────────────────────────────────────────────────────

def simulate_static(rows: list[dict]) -> list[int]:
    scaler    = StaticScaler()
    replicas  = []
    for row in rows:
        action = scaler.decide(row['cpu'], row['latency'])
        scaler.apply(action)
        replicas.append(scaler.replicas)
    return replicas


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACT AI REPLICA SERIES FROM CSV
# ─────────────────────────────────────────────────────────────────────────────

def extract_ai_replicas(rows: list[dict]) -> list[int]:
    """
    If the CSV has a 'replicas' column (what the AI system actually ran),
    use it directly as the 'AI controller' replay.
    """
    return [int(r['replicas']) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 30) -> str:
    filled = int(round(value / max(max_val, 1e-9) * width))
    return '█' * filled + '░' * (width - filled)


def print_report(ai: dict, static: dict, out_path: str | None = None):
    lines = []
    lines.append("=" * 62)
    lines.append("  BENCHMARK: Hybrid AI  vs  Static Threshold Scaler")
    lines.append("=" * 62)

    metrics_to_show = [
        ("SLA Violations (%)",     'sla_violation_pct', True),
        ("Mean Resource Waste",    'mean_waste',        True),
        ("Latency Mean (ms)",      'latency_mean_ms',   True),
        ("Latency Std-Dev (ms)",   'latency_stddev_ms', True),
        ("Mean Fail Ratio",        'mean_fail_ratio',   True),
        ("Mean Replicas",          'mean_replicas',     True),
    ]

    for label, key, lower_better in metrics_to_show:
        ai_val     = ai[key]
        static_val = static[key]
        max_val    = max(ai_val, static_val, 1e-9)

        lines.append(f"\n  {label}")
        lines.append(f"    Hybrid AI : {ai_val:>9.3f}  {_bar(ai_val, max_val)}")
        lines.append(f"    Static    : {static_val:>9.3f}  {_bar(static_val, max_val)}")

        if lower_better:
            winner = "Hybrid AI" if ai_val < static_val else \
                     "Static"   if static_val < ai_val else "Tie"
        else:
            winner = "Hybrid AI" if ai_val > static_val else \
                     "Static"   if static_val > ai_val else "Tie"
        lines.append(f"    → Better: {winner}")

    lines.append("\n" + "=" * 62)
    lines.append(f"  Total ticks evaluated: {ai['n_ticks']}")
    lines.append("=" * 62)

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
    parser = argparse.ArgumentParser(description="Benchmark AI vs Static scaler")
    parser.add_argument('--data', default='ai-controller/data/traffic_data.csv')
    parser.add_argument('--out',  default=None,
                        help='Optional path to write text report')
    parser.add_argument('--cap',  type=float, default=10.0,
                        help='Estimated capacity per server for waste calc')
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    rows = load_csv(args.data)
    if not rows:
        print("ERROR: No data loaded — check --data path.")
        return
    print(f"  {len(rows)} rows loaded.\n")

    # AI controller: use the replica counts that were actually running
    ai_replicas     = extract_ai_replicas(rows)
    static_replicas = simulate_static(rows)

    ai_metrics     = compute_metrics(rows, ai_replicas,     args.cap)
    static_metrics = compute_metrics(rows, static_replicas, args.cap)

    print_report(ai_metrics, static_metrics, out_path=args.out)


if __name__ == '__main__':
    main()