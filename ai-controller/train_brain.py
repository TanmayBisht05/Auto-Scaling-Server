import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from modules.predictor import LoadPredictor

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data/traffic_data.csv")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "models/load_predictor.pth")

# Match the Auto-Scaler's interval (5s)
# Window=10 means we look back 10 * 5s = 50 seconds of history
WINDOW_SIZE = 10
EPOCHS = 100
LEARNING_RATE = 0.001


def create_sequences(data, window_size):
    """
    Converts stream [10, 20, 30, 40, 50] into (Input, Label) pairs.
    """
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        seq = data[i: i + window_size]
        label = data[i + window_size]
        sequences.append(seq)
        labels.append(label)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def train():
    print(f"Starting Training using data at: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run collector.py first!")
        return

    # 1. Load Data (6 Columns now)
    df = pd.read_csv(DATA_PATH, header=None, names=[
                     'time', 'rps', 'cpu', 'replicas', 'latency', 'fail_ratio'])

    print(f"Loaded {len(df)} rows of raw traffic data.")

    # 2. Preprocessing: Calculate True Demand per row
    # We do this BEFORE resampling so we don't lose the "fail ratio" signal
    df['success_ratio'] = 1.0 - df['fail_ratio']
    df['success_ratio'] = df['success_ratio'].clip(lower=0.1)  # Safety
    df['true_demand'] = df['rps'] / df['success_ratio']

    # 3. Resample to 5-Second Buckets (The "Interpolating Step")
    # This matches your Auto-Scaler's decision interval
    df['datetime'] = pd.to_datetime(df['time'], unit='s', origin='unix')
    df = df.set_index('datetime')

    # We only care about 'true_demand' for the neural network training
    # We take the MEAN of the 5-second window
    df_resampled = df[['true_demand']].resample('5s').mean()

    # Fill gaps (Interpolate) so we have a continuous time series
    df_resampled = df_resampled.interpolate(method='linear')
    df_resampled = df_resampled.dropna()

    # Extract the numpy array
    raw_data = df_resampled['true_demand'].to_numpy(dtype=np.float32)

    print(f"Resampled to {len(raw_data)} data points (5s intervals).")

    # 4. Normalization (Min-Max Scaling)
    # Add 20% buffer to max so model can handle growth
    max_val = np.max(raw_data) * 1.2
    min_val = 0

    # Normalize and Clip
    normalized_data = (raw_data - min_val) / (max_val - min_val + 1e-6)
    normalized_data = np.clip(normalized_data, 0, 1)

    print(f"Training Stats: Max True RPS={max_val:.2f}")

    # 5. Prepare Tensors
    X, y = create_sequences(normalized_data, WINDOW_SIZE)

    if len(X) == 0:
        print(
            "Error: Not enough data after resampling! Run Locust for at least 60 seconds.")
        return

    # 6. Initialize Model
    model = LoadPredictor(input_size=WINDOW_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. Training Loop
    model.train()
    print("Training...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y.unsqueeze(1))
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

    # 8. Save Model & Metadata
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    torch.save({
        'model_state': model.state_dict(),
        'norm_max': max_val,
        'norm_min': min_val,
        'window_size': WINDOW_SIZE
    }, MODEL_SAVE_PATH)

    print(f"Model saved successfully to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
