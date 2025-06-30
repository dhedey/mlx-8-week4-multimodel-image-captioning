import torch.nn.functional as F
import torch.nn as nn
import torch
import einops
import math
from typing import Optional, Self
from .transformer import MaskedSelfAttentionConfig, MaskedSelfAttention, UnmaskedAttentionConfig, UnmaskedAttention, MultiLayerPerceptron
from ..common import ModuleConfig, Field

class DecoderBlockConfig(ModuleConfig):
    encoder_embedding_dimension: int
    decoder_embedding_dimension: int

    self_attention_kq_dimension: int
    self_attention_v_dimension: int
    self_attention_heads: int

    cross_attention_kq_dimension: int
    cross_attention_v_dimension: int
    cross_attention_heads: int

    mlp_hidden_dimension: int
    mlp_dropout: float

class DecoderBlock(nn.Module):
    def __init__(self, config: DecoderBlockConfig):
        super().__init__()
        embedding_dimension = config.decoder_embedding_dimension

        self.cross_attention = UnmaskedAttention(UnmaskedAttentionConfig(
            kq_dimension=config.cross_attention_kq_dimension,
            v_dimension=config.cross_attention_v_dimension,
            encoder_embedding_dimension=config.encoder_embedding_dimension,
            decoder_embedding_dimension=config.decoder_embedding_dimension,
            num_heads=config.cross_attention_heads
        ))
        self.ln_1 = nn.LayerNorm(embedding_dimension)
        self.masked_self_attention = MaskedSelfAttention(MaskedSelfAttentionConfig(
            kq_dimension=config.self_attention_kq_dimension,
            v_dimension=config.self_attention_v_dimension,
            embedding_dimension=embedding_dimension,
            num_heads=config.self_attention_heads,
        ))
        self.ln_2 = nn.LayerNorm(embedding_dimension)
        self.mlp = MultiLayerPerceptron(
            embedding_dimension=embedding_dimension,
            hidden_dimension=config.mlp_hidden_dimension,
            dropout=config.mlp_dropout,
        )
        self.ln_3 = nn.LayerNorm(embedding_dimension)

    def forward(self, encoder_embeddings, decoder_embeddings):
        residual_stream = decoder_embeddings
        residual_stream = residual_stream + self.cross_attention(encoder_embeddings, residual_stream)
        residual_stream = self.ln_1(residual_stream)
        residual_stream = residual_stream + self.masked_self_attention(residual_stream)
        residual_stream = self.ln_2(residual_stream)
        residual_stream = residual_stream + self.mlp(residual_stream)
        residual_stream = self.ln_3(residual_stream)
        return residual_stream

