import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from modules.predictor import LoadPredictor

# Configuration
DATA_PATH = "data/traffic_data.csv"
MODEL_SAVE_PATH = "models/load_predictor.pth"
WINDOW_SIZE = 5      # Look back 5 steps (25 seconds)
EPOCHS = 100         # How many times to loop through the data
LEARNING_RATE = 0.001

def create_sequences(data, window_size):
    """
    Converts a list [10, 20, 30, 40, 50, 60] into:
    X: [[10, 20, 30, 40, 50]]
    y: [60]
    """
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        seq = data[i : i + window_size]
        label = data[i + window_size]
        sequences.append(seq)
        labels.append(label)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

def train():
    print("Starting Training...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run generate_data.py first!")
        return

    df = pd.read_csv(DATA_PATH)
    rps_data = df['rps'].to_numpy(dtype=np.float32)
    
    # 2. Normalize Data (Crucial!)
    # We save these values because we need them to "un-normalize" later in the live system
    max_val = np.max(rps_data)
    min_val = np.min(rps_data)
    normalized_data = (rps_data - min_val) / (max_val - min_val)
    
    print(f"Data Stats: Max RPS={max_val}, Min RPS={min_val}")
    
    # 3. Prepare Tensors
    X, y = create_sequences(normalized_data, WINDOW_SIZE)
    
    # 4. Initialize Model
    model = LoadPredictor(input_size=WINDOW_SIZE)
    criterion = nn.MSELoss() # Mean Squared Error (Standard for regression)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y.unsqueeze(1)) # Compare prediction vs actual
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

    # 6. Save the Model
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # We save the weights AND the normalization params (we need them later!)
    torch.save({
        'model_state': model.state_dict(),
        'norm_max': max_val,
        'norm_min': min_val
    }, MODEL_SAVE_PATH)
    
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print("The Brain has learned to predict traffic!")

if __name__ == "__main__":
    train()