"""
Purpose: Neural Network training script. Ingests raw RPS time series data, resamples to 5-second windows, normalizes the data, and trains the PyTorch feedforward model to predict future load. Saves model state to disk.
Usage: Execute offline to generate the initial load_predictor.pth model, or trigger via the /retrain endpoint for online learning.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from modules.predictor import LoadPredictor

# --- CONFIGURATION ---
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH       = os.path.join(SCRIPT_DIR, "data/traffic_data.csv")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "models/load_predictor.pth")

WINDOW_SIZE   = 10
EPOCHS        = 100
LEARNING_RATE = 0.001


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADERS  (one per source format)
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_nasa_profile(profile_path: str) -> np.ndarray:
    """
    Load a NASA-derived profile CSV produced by nasa_traffic_profile.py.
    Expected columns: second_offset, rps
    Returns a float32 numpy array of RPS values (already at fixed intervals).
    """
    df = pd.read_csv(profile_path)

    # Accept both 'rps' header (new) and positional fallback
    if 'rps' in df.columns:
        raw = df['rps'].to_numpy(dtype=np.float32)
    elif df.shape[1] >= 2:
        raw = df.iloc[:, 1].to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Cannot find RPS column in {profile_path}")

    raw = np.nan_to_num(raw, nan=0.0)
    print(f"NASA profile loaded: {len(raw)} buckets  "
          f"(peak={raw.max():.1f} rps, mean={raw.mean():.1f} rps)")
    return raw


def _load_from_traffic_csv(data_path: str) -> np.ndarray:
    """
    Load traffic_data.csv (written by the live metrics collector).
    Expected columns: time, rps, cpu, replicas, latency, fail_ratio
    Returns true_demand resampled to 5-second buckets.
    """
    # Support both headered and header-less CSVs
    sample = pd.read_csv(data_path, nrows=1)
    if sample.columns[0].lower() in ('time', 'timestamp'):
        df = pd.read_csv(data_path)
        df.columns = [c.lower().strip() for c in df.columns]
    else:
        df = pd.read_csv(data_path, header=None,
                         names=['time', 'rps', 'cpu', 'replicas', 'latency', 'fail_ratio'])

    print(f"Loaded {len(df)} rows from {data_path}")

    fail_ratio = df['fail_ratio'] if 'fail_ratio' in df.columns else pd.Series(0.0, index=df.index)
    df['success_ratio'] = (1.0 - fail_ratio).clip(lower=0.1)
    df['true_demand']   = df['rps'] / df['success_ratio']

    # Resample to 5-second buckets (matches autoscaler interval)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', origin='unix')
    df = df.set_index('datetime')[['true_demand']].resample('5s').mean()
    df = df.interpolate(method='linear').dropna()

    raw = df['true_demand'].to_numpy(dtype=np.float32)
    print(f"Resampled to {len(raw)} 5-second data points.")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
#  SEQUENCE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def create_sequences(data: np.ndarray, window_size: int):
    sequences, labels = [], []
    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])
        labels.append(data[i + window_size])
    return (torch.tensor(sequences, dtype=torch.float32),
            torch.tensor(labels,    dtype=torch.float32))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(profile_path: str | None = None):
    """
    Train the load predictor.

    profile_path : path to a NASA-derived CSV  (second_offset, rps).
                   If None, falls back to traffic_data.csv.
    """
    # 1. Load raw RPS series ------------------------------------------------
    if profile_path:
        print(f"Training source: NASA profile  →  {profile_path}")
        if not os.path.exists(profile_path):
            print(f"ERROR: Profile not found at {profile_path}")
            return
        raw_data = _load_from_nasa_profile(profile_path)
    else:
        print(f"Training source: live traffic  →  {DATA_PATH}")
        if not os.path.exists(DATA_PATH):
            print(f"ERROR: {DATA_PATH} not found. Run collector first!")
            return
        raw_data = _load_from_traffic_csv(DATA_PATH)

    if len(raw_data) < WINDOW_SIZE + 1:
        print(f"ERROR: Need at least {WINDOW_SIZE + 1} data points, "
              f"got {len(raw_data)}.")
        return

    # 2. Normalisation ------------------------------------------------------
    # Add 20 % headroom so the model can extrapolate modest growth
    max_val = float(np.max(raw_data)) * 1.2
    min_val = 0.0

    norm = (raw_data - min_val) / (max_val - min_val + 1e-6)
    norm = np.clip(norm, 0, 1)

    print(f"Normalisation: min=0, max={max_val:.2f} (raw peak × 1.2)")

    # 3. Sequences ----------------------------------------------------------
    X, y = create_sequences(norm, WINDOW_SIZE)
    print(f"Sequences: {len(X)} training pairs  (window={WINDOW_SIZE})")

    # 4. Model --------------------------------------------------------------
    model     = LoadPredictor(input_size=WINDOW_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training loop ------------------------------------------------------
    model.train()
    print(f"Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X)
        loss    = criterion(outputs, y.unsqueeze(1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"   Epoch [{epoch+1:3d}/{EPOCHS}]  Loss: {loss.item():.6f}")

    # 6. Save ---------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'norm_max':    max_val,
        'norm_min':    min_val,
        'window_size': WINDOW_SIZE,
    }, MODEL_SAVE_PATH)
    print(f"Model saved  →  {MODEL_SAVE_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RPS load predictor")
    parser.add_argument(
        '--profile',
        default=None,
        metavar='PATH',
        help='Path to a NASA-derived nasa_profile_*.csv file. '
             'If omitted, trains on traffic_data.csv instead.'
    )
    args = parser.parse_args()
    train(profile_path=args.profile)