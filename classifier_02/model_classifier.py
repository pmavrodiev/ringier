import torch.nn as nn


class Multiclass(nn.Module):
    def __init__(self, input_dim: int, hidden_layer_1: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_1),
            nn.ReLU(),
            nn.Linear(hidden_layer_1, output_dim),
            

        )

        
        
    def forward(self, x):        
        return self.layers(x)