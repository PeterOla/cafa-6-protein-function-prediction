import torch
import torch.nn as nn

class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = [512, 256], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        curr_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
            
        # Output layer
        layers.append(nn.Linear(curr_dim, num_classes))
        # No Sigmoid here if we use BCEWithLogitsLoss
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
