"""
Purpose: Neural Network training script. Ingests raw RPS time series data,
resamples to 5-second windows, normalizes the data, and trains the PyTorch
feedforward model to predict future load. Saves model state to disk.

Two training sources are supported:
  1. A pre-generated traffic profile CSV (second_offset, rps) — used for
     initial offline training before any live data exists.
  2. A live traffic CSV written by collector.py — used for online retraining
     via the /retrain endpoint once real run data is available.

Usage:
    # Initial training from WorldCup profile (before any live run):
    python train_brain.py --profile ../locust/traffic_profile_train.csv

    # Retrain from a real AI run's recorded data:
    python train_brain.py --data data/ai_traffic_data.csv
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
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "models/load_predictor.pth")
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "data/ai_traffic_data.csv")

WINDOW_SIZE   = 10
EPOCHS        = 100
LEARNING_RATE = 0.001


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_traffic_profile(profile_path: str) -> np.ndarray:
    """
    Load a pre-generated traffic profile CSV (e.g. from traffic_profile.py).
    Expected columns: second_offset, rps
    Returns a float32 numpy array of RPS values.
    """
    df = pd.read_csv(profile_path)

    if 'rps' in df.columns:
        raw = df['rps'].to_numpy(dtype=np.float32)
    elif df.shape[1] >= 2:
        raw = df.iloc[:, 1].to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Cannot find RPS column in {profile_path}")

    raw = np.nan_to_num(raw, nan=0.0)
    print(f"Traffic profile loaded: {len(raw)} buckets  "
          f"(peak={raw.max():.1f} rps, mean={raw.mean():.1f} rps)")
    return raw


def _load_from_live_csv(data_path: str) -> np.ndarray:
    """
    Load a live traffic CSV written by collector.py.
    Expected columns (no header): time, rps, cpu, replicas, latency, fail_ratio
    where 'time' is elapsed seconds since collector start.
    Returns true_demand resampled to 5-second buckets.
    """
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

    # 'time' is elapsed seconds since collector start — convert to a relative
    # datetime so resample() works correctly. We use a fixed epoch as anchor
    # since only relative ordering matters, not the absolute timestamp.
    df['datetime'] = pd.to_datetime(df['time'], unit='s', origin=pd.Timestamp('2000-01-01'))
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

def train(profile_path: str | None = None,
          data_path:    str | None = None):
    """
    Train the load predictor.

    profile_path : path to a traffic profile CSV (second_offset, rps).
                   Use this for initial training before any live run exists.
    data_path    : path to a live collector CSV (ai_traffic_data.csv).
                   Use this for retraining on real observed traffic.

    If both are provided, profile_path takes priority.
    If neither is provided, falls back to DEFAULT_DATA_PATH.
    """
    if profile_path:
        print(f"Training source: traffic profile → {profile_path}")
        if not os.path.exists(profile_path):
            print(f"ERROR: Profile not found at {profile_path}")
            return
        raw_data = _load_from_traffic_profile(profile_path)

    else:
        path = data_path or DEFAULT_DATA_PATH
        print(f"Training source: live traffic   → {path}")
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. "
                  f"Run collector.py first or pass --profile for initial training.")
            return
        raw_data = _load_from_live_csv(path)

    if len(raw_data) < WINDOW_SIZE + 1:
        print(f"ERROR: Need at least {WINDOW_SIZE + 1} data points, "
              f"got {len(raw_data)}.")
        return

    # ── Normalisation ──────────────────────────────────────────────────────
    # Add 20% headroom above observed peak so the model can extrapolate
    # modest growth without immediately clipping to 1.0.
    max_val = float(np.max(raw_data)) * 1.2
    min_val = 0.0

    norm = (raw_data - min_val) / (max_val - min_val + 1e-6)
    norm = np.clip(norm, 0, 1)
    print(f"Normalisation: min=0, max={max_val:.2f} (raw peak × 1.2)")

    # ── Sequences ──────────────────────────────────────────────────────────
    X, y = create_sequences(norm, WINDOW_SIZE)
    print(f"Sequences: {len(X)} training pairs  (window={WINDOW_SIZE})")

    # ── Model ──────────────────────────────────────────────────────────────
    model     = LoadPredictor(input_size=WINDOW_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Training loop ──────────────────────────────────────────────────────
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

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'norm_max':    max_val,
        'norm_min':    min_val,
        'window_size': WINDOW_SIZE,
    }, MODEL_SAVE_PATH)
    print(f"Model saved → {MODEL_SAVE_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RPS load predictor")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--profile',
        default=None,
        metavar='PATH',
        help='Path to a traffic profile CSV (second_offset, rps) for initial '
             'offline training. Example: ../locust/traffic_profile_train.csv'
    )
    group.add_argument(
        '--data',
        default=None,
        metavar='PATH',
        help='Path to a live collector CSV for retraining on real data. '
             f'Defaults to {DEFAULT_DATA_PATH} if neither flag is given.'
    )

    args = parser.parse_args()
    train(profile_path=args.profile, data_path=args.data)