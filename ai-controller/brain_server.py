"""
Purpose: REST API inference engine. Loads the trained PyTorch predictor and optimized Fuzzy Logic parameters. Receives current system metrics, predicts future RPS, evaluates conditions through the fuzzy system, and returns a scaling action alongside panic overrides.
Usage: Run continuously to serve scaling decisions to the autoscaler agent.
"""

from modules.fuzzy_logic import FuzzyBrain
from modules.predictor import LoadPredictor
import os
import numpy as np
import json
import torch
from flask import Flask, request, jsonify
import threading
from train_brain import train

app = Flask(__name__)

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "models/load_predictor.pth")
PARAM_PATH  = os.path.join(SCRIPT_DIR, "models/fuzzy_params.json")

# Global State
history_rps = []
HISTORY_SIZE = 10

# ── Capacity estimator ───────────────────────────────────────────────────────
# Two separate alphas: shrink quickly under stress, recover more aggressively
# once conditions are healthy (asymmetry was the original bug — recovery was
# only allowed when load_per_server happened to *exceed* the current estimate,
# which almost never happens at low traffic).
ESTIMATED_CAPACITY_PER_SERVER = 10.0
ALPHA_SHRINK   = 0.25   # Fast response to overload
ALPHA_RECOVER  = 0.10   # Steady climb back to true capacity
CAPACITY_FLOOR = 1.0
CAPACITY_CEIL  = 200.0

# ── Brain init ───────────────────────────────────────────────────────────────
print("Loading Brain...")

# 1.  Neural Network
if os.path.exists(MODEL_PATH):
    checkpoint   = torch.load(MODEL_PATH, weights_only=False)
    norm_max     = checkpoint['norm_max']
    norm_min     = checkpoint['norm_min']
    HISTORY_SIZE = checkpoint.get('window_size', 10)

    predictor = LoadPredictor(input_size=HISTORY_SIZE)
    predictor.load_state_dict(checkpoint['model_state'])
    predictor.eval()
    print(f"Neural Network Loaded (Window={HISTORY_SIZE}, MaxRPS={int(norm_max)})")
else:
    print("WARNING: No model found! Run train_brain.py first.")
    norm_max  = 100
    norm_min  = 0
    predictor = None

# 2.  Fuzzy Logic — 6-param aware
def _load_fuzzy():
    """Build FuzzyBrain from saved params (or defaults) and return it."""
    if os.path.exists(PARAM_PATH):
        with open(PARAM_PATH, 'r') as f:
            bp = json.load(f)

        p_list = [
            bp.get('load_low',    0.50),
            bp.get('load_high',   0.80),
            bp.get('cpu_safe',   40.0),
            bp.get('cpu_danger', 80.0),
            bp.get('thresh_up',   0.45),   # now persisted by GA
            bp.get('thresh_down', -0.35),
        ]
        brain = FuzzyBrain(p_list)
        print("Optimized Fuzzy Parameters Loaded (6-param).")
    else:
        brain = FuzzyBrain()
        print("Using Default Fuzzy Parameters.")
    return brain

fuzzy = _load_fuzzy()
# Thresholds live inside the FuzzyBrain object — no separate globals needed
THRESH_UP, THRESH_DOWN = fuzzy.get_thresholds()


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/decide', methods=['POST'])
def decide():
    global history_rps, fuzzy, THRESH_UP, THRESH_DOWN

    data           = request.json
    curr_rps       = data.get('current_rps',  0)
    curr_cpu       = data.get('current_cpu',  0)
    replicas       = data.get('replicas',      1)
    curr_latency   = data.get('latency',       0)
    curr_fail_ratio = data.get('fail_ratio',   0)

    # ── 1. DYNAMIC CAPACITY LEARNING (symmetric) ─────────────────────────────
    success_ratio  = max(1.0 - curr_fail_ratio, 0.001)
    true_rps       = curr_rps / success_ratio

    LATENCY_PANIC    = 2000

    ESTIMATED_CAPACITY_PER_SERVER = 70.0

    # ── 2. PREDICTION ─────────────────────────────────────────────────────────
    history_rps.append(true_rps)
    if len(history_rps) > HISTORY_SIZE:
        history_rps.pop(0)

    pred_rps = true_rps   # default: use current as prediction
    if predictor and len(history_rps) == HISTORY_SIZE:
        input_seq  = np.array(history_rps, dtype=np.float32)
        input_norm = (input_seq - norm_min) / (norm_max - norm_min + 1e-6)
        input_norm = np.clip(input_norm, 0, 1)

        with torch.no_grad():
            pred_norm = predictor(torch.tensor(input_norm).unsqueeze(0)).item()
        pred_rps = pred_norm * (norm_max - norm_min) + norm_min

    # ── 3. FUZZY DECISION ────────────────────────────────────────────────────
    total_capacity = replicas * ESTIMATED_CAPACITY_PER_SERVER
    load_ratio     = pred_rps / max(total_capacity, 0.1)

    score  = fuzzy.compute(load_ratio, curr_cpu)
    action = "HOLD"

    if score > THRESH_UP:
        action = "SCALE_UP"
    elif score < THRESH_DOWN:
        action = "SCALE_DOWN"

    # ── 4. PANIC OVERRIDE ────────────────────────────────────────────────────
    is_dying = False

    if curr_latency > LATENCY_PANIC:
        action   = "SCALE_UP"
        score    = 2.0
        is_dying = True

    if curr_cpu > 95.0:
        action   = "SCALE_UP"
        score    = 2.0
        is_dying = True

    if is_dying:
        print("!!! PANIC MODE TRIGGERED: Scaling Up !!!")

    return jsonify({
        "action":             action,
        "predicted_rps":      int(pred_rps),
        "fuzzy_score":        round(score, 3),
        "estimated_capacity": round(ESTIMATED_CAPACITY_PER_SERVER, 2),
        "true_demand":        int(true_rps),
        "load_ratio":         round(load_ratio, 3),
        "latency":            int(curr_latency),
        "thresh_up":          round(THRESH_UP, 3),
        "thresh_down":        round(THRESH_DOWN, 3),
    })


@app.route('/reload_params', methods=['POST'])
def reload_params():
    """Hot-reload fuzzy params after the GA writes a new fuzzy_params.json."""
    global fuzzy, THRESH_UP, THRESH_DOWN
    fuzzy = _load_fuzzy()
    THRESH_UP, THRESH_DOWN = fuzzy.get_thresholds()
    return jsonify({"status": "reloaded", "thresh_up": THRESH_UP, "thresh_down": THRESH_DOWN})


@app.route('/retrain', methods=['POST'])
def trigger_retrain():
    def run_job():
        print("Background Retraining started...")
        train()
        print("Retraining complete. Restart server to apply.")
    threading.Thread(target=run_job, daemon=True).start()
    return jsonify({"status": "training_started"})


from optimizer import run_online_ga
run_online_ga(interval=120, window=200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)