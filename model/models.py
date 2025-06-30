import torch.nn.functional as F
import torch.nn as nn
import torch
import re
import os
import statistics
import transformers
import random
import einops
import pandas as pd
import math
from typing import Optional, Self
from .common import ModelBase, ModuleConfig, TrainingConfig, Field
from .modules.encoder import EncoderBlockConfig, ImageEncoder, ImageEncoderConfig
from .modules.decoder import DecoderBlockConfig, DecoderBlock

class SingleDigitModelConfig(ModuleConfig):
    encoder: ImageEncoderConfig

class SingleDigitModel(ModelBase):
    def __init__(self, model_name: str, config: SingleDigitModelConfig):
        super().__init__(model_name=model_name, config=config)
        self.image_encoder = ImageEncoder(config.encoder)
        self.prediction_layer = nn.Linear(config.encoder.embedding_dimension, 10)  # 10 digits (0-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Assumes it receives a (Batch(es), ChannelCount=1, Height=28, Width=28) tensor of images."""

        residual_stream=self.image_encoder(x)
        averaged_embedding = residual_stream.mean(dim=-2)   # Average over the blocks, Shape: (Batch(es), Embedding Dimension)
        logits = self.prediction_layer(averaged_embedding)  # Shape: (Batch(es), 10)

        return logits

class DigitSequenceModelConfig(ModuleConfig):
    encoder: ImageEncoderConfig
    max_sequence_length: int
    decoder_block_count: int
    decoder_block: DecoderBlockConfig

class DigitSequenceModel(ModelBase):
    def __init__(self, model_name: str, config: DigitSequenceModelConfig):
        super().__init__(model_name=model_name, config=config)

        # It's already stored in the base class. But this helps the IDE understand its type.
        self.config = config

        if config.decoder_block.encoder_embedding_dimension != config.encoder.embedding_dimension:
            raise ValueError("Config inconsistency: The encoder embedding dimension in the encoder and decoder disagree")

        self.image_encoder = ImageEncoder(config.encoder)

        decoder_embedding_dimension = config.decoder_block.decoder_embedding_dimension

        # 12 embedding tokens: 10 digits, 1 padding token, 1 start token
        self.digit_sequence_embedding = nn.Embedding(
            num_embeddings=12,
            embedding_dim=decoder_embedding_dimension,
            padding_idx=0,
        )
        self.digit_sequence_positional_bias = nn.Parameter(
            torch.zeros([config.max_sequence_length, decoder_embedding_dimension], dtype=torch.float32),
        )
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(config.decoder_block) for _ in range(config.decoder_block_count)
        ])

        self.decoder_prediction_layer = nn.Linear(decoder_embedding_dimension, 11)

    def forward(self, image, sequence):
        # image    => Shape: (batch_size, image_width, image_height)
        # sequence => Shape: (batch_size, decoder_sequence_length)
        #add batch dimension if it's missing
        if image.dim() == 2:
            image = image.unsqueeze(0)

        if image.dim() == 3:
            # We add a singleton channel dimension if it's missing, so the image encoder works
            image = image.unsqueeze(1)

        encoded_image = self.image_encoder(image)
            
        # We add 1 to the sequence indices to account for the padding token we inserted at index 0
        embedded_sequence = self.digit_sequence_embedding(sequence + 1)
        residual_stream = embedded_sequence + self.digit_sequence_positional_bias

        for decoder_block in self.decoder_blocks:
            residual_stream = decoder_block(encoded_image, residual_stream)

        logits = self.decoder_prediction_layer(residual_stream)
        
        return logits
        
if __name__ == "__main__":
   print("Run default_models instead of this file")