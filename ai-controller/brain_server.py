from modules.fuzzy_logic import FuzzyBrain
from modules.predictor import LoadPredictor
import os
import numpy as np
import json
import torch
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- 1. LOAD THE BRAIN ---
print("Loading Brain Models...")

# A. Load Neural Network
model_path = "models/load_predictor.pth"
# Set weights_only=False because we trust our own file
checkpoint = torch.load(model_path, weights_only=False)

predictor = LoadPredictor()
predictor.load_state_dict(checkpoint['model_state'])
predictor.eval()  # Set to evaluation mode

# Load Normalization Params (Crucial!)
# We renamed these to match the global scope usage below
norm_max = checkpoint['norm_max']
norm_min = checkpoint['norm_min']

# B. Load Fuzzy Logic
param_path = "models/fuzzy_params.json"
if os.path.exists(param_path):
    with open(param_path, 'r') as f:
        best_params = json.load(f)
    # Convert dict to list [low, high, safe, danger]
    p_list = [best_params['load_low'], best_params['load_high'],
              best_params['cpu_safe'], best_params['cpu_danger']]
    fuzzy = FuzzyBrain(p_list)
    print("Optimized Fuzzy Logic Loaded!")
else:
    fuzzy = FuzzyBrain()  # Use defaults if GA didn't run
    print("Warning: Optimized params not found. Using defaults.")

# State Memory (We need the last 5 rps to make a prediction)
history_rps = []


@app.route('/decide', methods=['POST'])
def decide():
    """
    Input JSON: { "current_rps": 500, "current_cpu": 45, "replicas": 5 }
    """
    global history_rps

    data = request.json
    curr_rps = data.get('current_rps', 0)
    curr_cpu = data.get('current_cpu', 0)
    replicas = data.get('replicas', 5)

    # 1. Update History
    history_rps.append(curr_rps)
    if len(history_rps) > 5:
        history_rps.pop(0)

    # If we don't have enough data yet, just hold
    if len(history_rps) < 5:
        return jsonify({"action": "HOLD", "reason": "Not enough data"})

    # 2. Neural Network Prediction
    # Normalize input
    input_seq = np.array(history_rps, dtype=np.float32)
    # Use global norm_max/norm_min
    input_norm = (input_seq - norm_min) / (norm_max - norm_min)
    input_tensor = torch.tensor(input_norm).unsqueeze(0)  # Add batch dim

    with torch.no_grad():
        pred_norm = predictor(input_tensor).item()

    # Denormalize output (FIXED HERE)
    pred_rps = pred_norm * (norm_max - norm_min) + norm_min

    # 3. Fuzzy Decision
    capacity_per_server = 150  # Assumption
    total_capacity = replicas * capacity_per_server

    # Avoid division by zero
    if total_capacity == 0:
        total_capacity = 150

    load_ratio = pred_rps / total_capacity

    decision_score = fuzzy.compute(load_ratio, curr_cpu)

    # 4. Final Policy
    action = "HOLD"
    if decision_score > 0.6:
        action = "SCALE_UP"
    if decision_score < -0.6:
        action = "SCALE_DOWN"

    return jsonify({
        "action": action,
        "predicted_rps": int(pred_rps),
        "fuzzy_score": round(decision_score, 2)
    })


if __name__ == '__main__':
    # Run on port 6000 so it doesn't conflict with his Node app (5000)
    app.run(host='0.0.0.0', port=6000)
