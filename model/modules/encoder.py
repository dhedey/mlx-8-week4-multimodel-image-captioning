import torch.nn.functional as F
import torch.nn as nn
import torch
import einops
import math
from typing import Optional, Self
from .transformer import UnmaskedAttentionConfig, UnmaskedAttention, MultiLayerPerceptron
from ..common import ModuleConfig, Field

class EncoderBlockConfig(ModuleConfig):
    kq_dimension: int
    v_dimension: int
    embedding_dimension: int
    num_heads: int
    mlp_hidden_dimension: Optional[int]
    mlp_dropout: float

class EncoderBlock(nn.Module):
    def __init__(self, config: EncoderBlockConfig):
        super().__init__()

        attention_config = UnmaskedAttentionConfig(
            kq_dimension=config.kq_dimension,
            v_dimension=config.v_dimension,
            encoder_embedding_dimension=config.embedding_dimension,
            decoder_embedding_dimension=config.embedding_dimension,
            num_heads=config.num_heads,
        )
        self.attention = UnmaskedAttention(attention_config)

        self.layer_norm_1 = nn.LayerNorm(config.embedding_dimension)

        mlp_hidden_dimension = config.mlp_hidden_dimension

        if mlp_hidden_dimension is None:
            mlp_hidden_dimension = config.embedding_dimension * 4 # Commonly used in transformers

        self.mlp = MultiLayerPerceptron(
            embedding_dimension=config.embedding_dimension,
            hidden_dimension=mlp_hidden_dimension,
            dropout=config.mlp_dropout,
        )
        self.layer_norm_2 = nn.LayerNorm(config.embedding_dimension)

    def forward(self, residual_stream):
        # Shape: (Batch, Sequence Length, Embedding Dimension)
        residual_stream = residual_stream + self.attention(residual_stream, residual_stream)
        residual_stream = self.layer_norm_1(residual_stream)
        residual_stream = residual_stream + self.mlp(residual_stream)         
        residual_stream = self.layer_norm_2(residual_stream)

        return residual_stream
    
class PatchEmbedder(nn.Module):
    def __init__(
            self,
            image_width: int,
            image_height: int,
            patch_width: int,
            patch_height: int,
            embedding_dimension: int,
        ):
        super().__init__()
        self.patch_width = patch_width
        self.patch_height = patch_height

        vertical_patches = image_height // patch_height
        horizontal_patches = image_width // patch_width
        total_patches = vertical_patches * horizontal_patches

        self.patch_embedding = nn.Linear(patch_width * patch_height, embedding_dimension)
        self.patch_positional_bias = nn.Parameter(
            torch.zeros([total_patches, embedding_dimension], dtype=torch.float32),
        )

    def forward(self, image):
        flattened_image_patches = einops.rearrange(
            image,
            'batch 1 (h_patches patch_height) (w_patches patch_width) -> batch (h_patches w_patches) (patch_height patch_width)',
            patch_width=self.patch_width,
            patch_height=self.patch_height
        ) # Shape: (batch_size, channel, total_patches, flattened_patch_size)

        embedded_patches = self.patch_embedding(flattened_image_patches) # Shape: (batch_size, encoder_blocks, encoder_embedding_size)

        return embedded_patches + self.patch_positional_bias


class ImageEncoderConfig(ModuleConfig):
    image_width: int
    image_height: int
    image_patch_width: int
    image_patch_height: int
    embedding_dimension: int
    encoder_block_count: int
    encoder_block: EncoderBlockConfig
    
    def __post_init__(self):
        if self.image_width % self.image_patch_width != 0 or self.image_height % self.image_patch_height != 0:
            raise ValueError("Image dimensions must be divisible by block dimensions.")

class ImageEncoder(nn.Module):
    def __init__(self, config: ImageEncoderConfig):
        super().__init__()
        self.patch_embedder = PatchEmbedder(
            image_width=config.image_width,
            image_height=config.image_height,
            patch_width=config.image_patch_width,
            patch_height=config.image_patch_height,
            embedding_dimension=config.embedding_dimension,
        )

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(config.encoder_block)
            for _ in range(config.encoder_block_count)
        ])

    def forward(self, image):
        residual_stream = self.patch_embedder(image) # Shape: (batch_size, patches, embedding_size)

        for encoder_block in self.encoder_blocks:
            residual_stream = encoder_block(residual_stream)

        return residual_stream
    