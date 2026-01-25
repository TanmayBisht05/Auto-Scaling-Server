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
MODEL_PATH = os.path.join(SCRIPT_DIR, "models/load_predictor.pth")
PARAM_PATH = os.path.join(SCRIPT_DIR, "models/fuzzy_params.json")

# Global State
history_rps = []
HISTORY_SIZE = 10
ESTIMATED_CAPACITY_PER_SERVER = 10.0
ALPHA = 0.2

# --- LOAD BRAIN ---
print("Loading Brain...")

# 1. Load Neural Network
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    norm_max = checkpoint['norm_max']
    norm_min = checkpoint['norm_min']
    HISTORY_SIZE = checkpoint.get('window_size', 10)

    predictor = LoadPredictor(input_size=HISTORY_SIZE)
    predictor.load_state_dict(checkpoint['model_state'])
    predictor.eval()
    print(
        f"Neural Network Loaded (Window={HISTORY_SIZE}, MaxRPS={int(norm_max)})")
else:
    print("WARNING: No model found! Run train_brain.py first.")
    norm_max = 100
    norm_min = 0
    predictor = None

# 2. Load Fuzzy Logic
THRESH_UP = 0.6
THRESH_DOWN = -0.6

if os.path.exists(PARAM_PATH):
    with open(PARAM_PATH, 'r') as f:
        best_params = json.load(f)

    p_list = [
        best_params.get('load_low', 0.5),
        best_params.get('load_high', 0.8),
        best_params.get('cpu_safe', 40),
        best_params.get('cpu_danger', 80)
    ]
    fuzzy = FuzzyBrain(p_list)
    THRESH_UP = best_params.get('thresh_up', 0.6)
    THRESH_DOWN = best_params.get('thresh_down', -0.6)
    print("Optimized Fuzzy Parameters Loaded!")
else:
    fuzzy = FuzzyBrain()
    print("Using Default Fuzzy Parameters.")


@app.route('/decide', methods=['POST'])
def decide():
    global history_rps, ESTIMATED_CAPACITY_PER_SERVER

    data = request.json
    curr_rps = data.get('current_rps', 0)
    curr_cpu = data.get('current_cpu', 0)
    replicas = data.get('replicas', 1)
    curr_latency = data.get('latency', 0)
    curr_fail_ratio = data.get('fail_ratio', 0)

    # --- 1. DYNAMIC CAPACITY LEARNING ---
    success_ratio = max(1.0 - curr_fail_ratio, 0.001)
    true_rps = curr_rps / success_ratio
    load_per_server = true_rps / max(replicas, 1)

    # Thresholds
    LATENCY_PANIC = 2000   # 2.0 Seconds
    LATENCY_SAFE = 400     # 400ms

    # Capacity Adjustment
    if curr_latency > LATENCY_PANIC or curr_fail_ratio > 0.01:
        if ESTIMATED_CAPACITY_PER_SERVER > load_per_server:
            ESTIMATED_CAPACITY_PER_SERVER = (
                1 - ALPHA) * ESTIMATED_CAPACITY_PER_SERVER + ALPHA * load_per_server

    elif curr_latency < LATENCY_SAFE and curr_fail_ratio < 0.001:
        if load_per_server > ESTIMATED_CAPACITY_PER_SERVER:
            ESTIMATED_CAPACITY_PER_SERVER = (
                1 - ALPHA) * ESTIMATED_CAPACITY_PER_SERVER + ALPHA * load_per_server

    ESTIMATED_CAPACITY_PER_SERVER = max(
        1.0, min(ESTIMATED_CAPACITY_PER_SERVER, 100.0))

    # --- 2. PREDICTION ---
    history_rps.append(true_rps)
    if len(history_rps) > HISTORY_SIZE:
        history_rps.pop(0)

    pred_rps = true_rps
    if predictor and len(history_rps) == HISTORY_SIZE:
        input_seq = np.array(history_rps, dtype=np.float32)
        input_norm = (input_seq - norm_min) / (norm_max - norm_min + 1e-6)
        input_norm = np.clip(input_norm, 0, 1)

        with torch.no_grad():
            pred_norm = predictor(torch.tensor(input_norm).unsqueeze(0)).item()
        pred_rps = pred_norm * (norm_max - norm_min) + norm_min

    # --- 3. FUZZY DECISION ---
    total_capacity = replicas * ESTIMATED_CAPACITY_PER_SERVER
    load_ratio = pred_rps / max(total_capacity, 0.1)

    score = fuzzy.compute(load_ratio, curr_cpu)

    action = "HOLD"
    if score > THRESH_UP:
        action = "SCALE_UP"
    elif score < THRESH_DOWN:
        action = "SCALE_DOWN"

    # --- 4. PANIC OVERRIDE (The Fix) ---
    # If the system is dying, ignore the brain and scream for help.
    is_dying = False

    # Condition A: Latency is unbearable (> 2000ms)
    if curr_latency > LATENCY_PANIC:
        action = "SCALE_UP"
        score = 2.0  # Force max score
        is_dying = True

    # Condition B: CPU is pegged (> 95%) AND we are not at max scale
    if curr_cpu > 95.0:
        action = "SCALE_UP"
        score = 2.0
        is_dying = True

    # Prevent "Death Spiral" where 0 RPS makes us hold
    if is_dying:
        print("!!! PANIC MODE TRIGGERED: Scaling Up !!!")

    return jsonify({
        "action": action,
        "predicted_rps": int(pred_rps),
        "fuzzy_score": round(score, 2),
        "estimated_capacity": round(ESTIMATED_CAPACITY_PER_SERVER, 2),
        "true_demand": int(true_rps),
        "latency": int(curr_latency)
    })


@app.route('/retrain', methods=['POST'])
def trigger_retrain():
    def run_job():
        print("Background Retraining started...")
        train()
        print("Retraining complete. Restart server to apply.")
    thread = threading.Thread(target=run_job)
    thread.start()
    return jsonify({"status": "training_started"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
