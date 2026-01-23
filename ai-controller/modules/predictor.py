import torch
import torch.nn as nn

class LoadPredictor(nn.Module):
    def __init__(self, input_size=5):
        super(LoadPredictor, self).__init__()
        
        # We use a simple 3-layer network (Input -> Hidden -> Output)
        # It's fast, stable, and perfect for real-time control.
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),  # Layer 1: Takes 5 past points, expands to 32 features
            nn.ReLU(),                  # Activation: Adds non-linearity
            
            nn.Linear(32, 16),          # Layer 2: Compresses features to 16
            nn.ReLU(),
            
            nn.Linear(16, 1)            # Output Layer: Predicts the single next value
        )
    
    def forward(self, x):
        return self.net(x)