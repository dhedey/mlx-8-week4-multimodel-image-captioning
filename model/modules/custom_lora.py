import torch.nn.functional as F
import torch.nn as nn
import torch
import einops
import math
from typing import Optional, Self
from ..common import ModuleConfig, Field

class CustomLoraConfig(ModuleConfig):
    input_dim: int
    output_dim: int
    rank: int
    dropout: float = 0.1

class CustomLora(nn.Module):
    """
    A simple low-rank linear layer with dropout.
    W = A @ B where A and B have low rank.
    """
    def __init__(self, config: CustomLoraConfig):
        super().__init__()
        self.config = config
        self.rank = config.rank
        self.dropout = nn.Dropout(config.dropout)
        
        # Low-rank decomposition: W = A @ B
        # A: (input_dim, rank), B: (rank, output_dim)
        self.A = nn.Linear(config.input_dim, config.rank, bias=False)
        self.B = nn.Linear(config.rank, config.output_dim, bias=True)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.A(x)  # (batch_size, seq_len, rank)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.B(x)  # (batch_size, seq_len, output_dim)
        return x



