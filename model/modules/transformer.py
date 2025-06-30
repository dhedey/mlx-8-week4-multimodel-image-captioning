import torch.nn.functional as F
import torch.nn as nn
import torch
import einops
import math
from typing import Optional, Self
from ..common import ModuleConfig, Field

class MultiLayerPerceptron(nn.Module):
    def __init__(self, embedding_dimension: int, hidden_dimension: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, embedding_dimension)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x has            Shape: (batch_size, sequence_length, embedding_dimension)
        x = self.fc1(x)  # Shape: (batch_size, sequence_length, hidden_dimension)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Shape: (batch_size, sequence_length, embedding_dimension)
        return x

class UnmaskedAttentionConfig(ModuleConfig):
    kq_dimension: int
    v_dimension: int
    encoder_embedding_dimension: int
    decoder_embedding_dimension: int
    num_heads: int

# TODO: Combine this into UnmaskedAttentionHead
class UnmaskedAttention(nn.Module):
    def __init__(self, params: UnmaskedAttentionConfig):
        super().__init__()
        self.heads = nn.ModuleList([
            UnmaskedAttentionHead(
                kq_dimension=params.kq_dimension,
                v_dimension=params.v_dimension,
                encoder_embedding_dimension=params.encoder_embedding_dimension,
                decoder_embedding_dimension=params.decoder_embedding_dimension,
            )
            for _ in range(params.num_heads)
        ])
    
    def forward(self, encoder_embeddings, decoder_embeddings):
        # encoder_embeddings has shape (batch_size, encoder_sequence_length, embedding_dimension)
        # decoder_embeddings has shape (batch_size, decoder_sequence_length, embedding_dimension)
        residual_sum = torch.zeros_like(decoder_embeddings)
        for head in self.heads:
            residual_sum = residual_sum + head(encoder_embeddings, decoder_embeddings)
        return residual_sum
    
class UnmaskedAttentionHead(nn.Module):
    def __init__(self, kq_dimension, v_dimension, encoder_embedding_dimension, decoder_embedding_dimension):
        super().__init__()
        self.kq_dimension = kq_dimension
        self.v_dimension = v_dimension
        self.encoder_embedding_dimension = encoder_embedding_dimension
        self.decoder_embedding_dimension = decoder_embedding_dimension

        self.query_projection = nn.Linear(decoder_embedding_dimension, kq_dimension)
        self.key_projection = nn.Linear(encoder_embedding_dimension, kq_dimension)
        self.value_projection = nn.Linear(encoder_embedding_dimension, v_dimension)
        
        # NB - Unlike many transformer implementations, we have a per-head output projection for easy of understanding.
        self.output_projection = nn.Linear(v_dimension, decoder_embedding_dimension)

    def forward(self, encoder_embeddings, decoder_embeddings):
        queries = self.query_projection(decoder_embeddings)  # Shape: (batch_size, decoder_sequence_length, kq_dimension)
        keys = self.key_projection(encoder_embeddings)       # Shape: (batch_size, encoder_sequence_length, kq_dimension)
        values = self.value_projection(encoder_embeddings)   # Shape: (batch_size, encoder_sequence_length, v_dimension)

        attention_scores = queries @ keys.transpose(-2, -1)  # Shape: (batch_size, decoder_sequence_length, encoder_sequence_length)

        # Softmax over the encoder_sequence_length (keys), so each query row sums to 1
        attention = F.softmax(attention_scores / math.sqrt(self.kq_dimension), dim=-1)

        output_values = attention @ values                   # Shape: (batch_size, decoder_sequence_length, v_dimension)
        
        residual = self.output_projection(output_values)     # Shape: (batch_size, decoder_sequence_length, decoder_embedding_dimension)
        return residual

class MaskedSelfAttentionConfig(ModuleConfig):
    kq_dimension: int
    v_dimension: int
    embedding_dimension: int
    num_heads: int

class MaskedSelfAttention(nn.Module):
    def __init__(self, params: MaskedSelfAttentionConfig, mask_future_tokens=True):
        super().__init__()
        self.embedding_dimension = params.embedding_dimension
        self.kq_dimension = params.kq_dimension
        self.v_dimension = params.v_dimension
        self.num_heads = params.num_heads

        self.query_projection = nn.Linear(params.embedding_dimension, params.kq_dimension * params.num_heads)
        self.key_projection = nn.Linear(params.embedding_dimension, params.kq_dimension * params.num_heads)
        self.value_projection = nn.Linear(params.embedding_dimension, params.v_dimension * params.num_heads)
        self.output_projection = nn.Linear(params.v_dimension * params.num_heads, params.embedding_dimension)
        self.mask_future_tokens = mask_future_tokens

        self.register_buffer('_cached_mask', torch.empty(0), persistent=False)


    def forward(self, x): #x has shape (batch_size, sequence_length, embedding_dimension)
        queries = self.query_projection(x).reshape(*x.shape[:-1], self.num_heads, self.kq_dimension).transpose(-2, -3)  # Shape: (batch_size, num_heads, sequence_length, kq_dimension)
        keys = self.key_projection(x).reshape(*x.shape[:-1], self.num_heads, self.kq_dimension).transpose(-2, -3)       # Shape: (batch_size, num_heads, sequence_length, kq_dimension)
        values = self.value_projection(x).reshape(*x.shape[:-1], self.num_heads, self.v_dimension).transpose(-2, -3)   # Shape: (batch_size, num_heads, sequence_length, v_dimension)

        attention_scores = queries @ keys.transpose(-2, -1) # Shape: (batch_size, num_heads, sequence_length, sequence_length)
        if self.mask_future_tokens:
            seq_len = x.shape[-2]
            if self._cached_mask.shape[-1] < seq_len:
                mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device), diagonal=1)
                self._cached_mask = mask
            
            # Since self._cached_mask is on the same device as the model, we can use it directly
            attention_scores = attention_scores.masked_fill(self._cached_mask[:seq_len, :seq_len], float("-inf"))

        attention = F.softmax(attention_scores / math.sqrt(self.kq_dimension), dim=-1)
        output_values = attention @ values                   # Shape: (batch_size, num_heads, sequence_length, v_dimension)
        output_values = output_values.transpose(-2, -3).reshape(*x.shape[:-1], self.num_heads * self.v_dimension) # Shape: (batch_size, sequence_length, num_heads * v_dimension)
        residual = self.output_projection(output_values)     # Shape: (batch_size, sequence_length, embedding_dimension)
        return residual