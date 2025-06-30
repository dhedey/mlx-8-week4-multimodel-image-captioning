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
from .models import DigitSequenceModel, DigitSequenceModelConfig, SingleDigitModel, SingleDigitModelConfig
from .trainer import ModelTrainerBase, DigitSequenceModelTrainer, DigitSequenceModelTrainingConfig, SingleDigitModelTrainer, SingleDigitModelTrainingConfig
from .wandb_config import WANDB_ENTITY, WANDB_PROJECT_NAME

DEFAULT_MODEL_PARAMETERS = {
    "multi-digit-scrambled-v2": {
        "model_class": DigitSequenceModel,
        "model": DigitSequenceModelConfig(
            max_sequence_length=10,
            encoder=ImageEncoderConfig(
                image_width=70,
                image_height=70,
                image_patch_width=7,
                image_patch_height=7,
                embedding_dimension=64,
                encoder_block_count=7,
                encoder_block=EncoderBlockConfig(
                    kq_dimension=32,
                    v_dimension=32,
                    embedding_dimension=64,
                    num_heads=8,
                    mlp_hidden_dimension=256,
                    mlp_dropout=0.2,
                ),
            ),
            decoder_block_count=5,
            decoder_block=DecoderBlockConfig(
                encoder_embedding_dimension=64,
                decoder_embedding_dimension=64,

                self_attention_kq_dimension=32,
                self_attention_v_dimension=32,
                self_attention_heads=8,

                cross_attention_kq_dimension=32,
                cross_attention_v_dimension=32,
                cross_attention_heads=8,

                mlp_hidden_dimension=256,
                mlp_dropout=0.2,
            ),
        ),
        "model_trainer": DigitSequenceModelTrainer,
        "training": DigitSequenceModelTrainingConfig(
            batch_size=128,
            epochs=20,
            learning_rate=0.00015,
            training_set_size=30000,
            validation_set_size=5000,
            optimizer="adamw",
            generator_kind="david-v2",
        ),
    },
    "multi-digit-scrambled": {
        "model_class": DigitSequenceModel,
        "model": DigitSequenceModelConfig(
            max_sequence_length=5 * 5 + 1,
            encoder=ImageEncoderConfig(
                image_width=28 * 5,
                image_height=28 * 5,
                image_patch_width=7,
                image_patch_height=7,
                embedding_dimension=64,
                encoder_block_count=10,
                encoder_block=EncoderBlockConfig(
                    kq_dimension=32,
                    v_dimension=32,
                    embedding_dimension=64,
                    num_heads=8,
                    mlp_hidden_dimension=256,
                    mlp_dropout=0.2,
                ),
            ),
            decoder_block_count=5,
            decoder_block=DecoderBlockConfig(
                encoder_embedding_dimension=64,
                decoder_embedding_dimension=64,

                self_attention_kq_dimension=32,
                self_attention_v_dimension=32,
                self_attention_heads=8,

                cross_attention_kq_dimension=32,
                cross_attention_v_dimension=32,
                cross_attention_heads=8,

                mlp_hidden_dimension=256,
                mlp_dropout=0.2,
            ),
        ),
        "model_trainer": DigitSequenceModelTrainer,
        "training": DigitSequenceModelTrainingConfig(
            batch_size=64, # 256 took too much GPU memory
            epochs=20,
            learning_rate=0.00015,
            optimizer="adamw",
            generator_kind="nick",
        ),
    },
    "multi-digit-v2": {
        "model_class": DigitSequenceModel,
        "model": DigitSequenceModelConfig(
            max_sequence_length=4 * 4 + 1,
            encoder=ImageEncoderConfig(
                image_width=28 * 4,
                image_height=28 * 4,
                image_patch_width=7,
                image_patch_height=7,
                embedding_dimension=64,
                encoder_block_count=5,
                encoder_block=EncoderBlockConfig(
                    kq_dimension=32,
                    v_dimension=32,
                    embedding_dimension=64,
                    num_heads=2,
                    mlp_hidden_dimension=256,
                    mlp_dropout=0.2,
                ),
            ),
            decoder_block_count=2,
            decoder_block=DecoderBlockConfig(
                encoder_embedding_dimension=64,
                decoder_embedding_dimension=64,

                self_attention_kq_dimension=32,
                self_attention_v_dimension=32,
                self_attention_heads=2,

                cross_attention_kq_dimension=32,
                cross_attention_v_dimension=32,
                cross_attention_heads=2,

                mlp_hidden_dimension=256,
                mlp_dropout=0.2,
            ),
        ),
        "model_trainer": DigitSequenceModelTrainer,
        "training": DigitSequenceModelTrainingConfig(
            batch_size=256,
            epochs=20,
            learning_rate=0.0002,
            optimizer="adamw",
            warmup_epochs=5,
        ),
    },
    "multi-digit-v1": {
        "model_class": DigitSequenceModel,
        "model": DigitSequenceModelConfig(
            max_sequence_length=11,
            encoder=ImageEncoderConfig(
                image_width=56,
                image_height=56,
                image_patch_width=7,
                image_patch_height=7,
                embedding_dimension=32,
                encoder_block_count=5,
                encoder_block=EncoderBlockConfig(
                    kq_dimension=16,
                    v_dimension=16,
                    embedding_dimension=32,
                    num_heads=2,
                    mlp_hidden_dimension=128,
                    mlp_dropout=0.2,
                ),
            ),
            decoder_block_count=2,
            decoder_block=DecoderBlockConfig(
                encoder_embedding_dimension=32,
                decoder_embedding_dimension=32,

                self_attention_kq_dimension=16,
                self_attention_v_dimension=16,
                self_attention_heads=2,

                cross_attention_kq_dimension=16,
                cross_attention_v_dimension=16,
                cross_attention_heads=2,

                mlp_hidden_dimension=128,
                mlp_dropout=0.2,
            ),
        ),
        "model_trainer": DigitSequenceModelTrainer,
        "training": DigitSequenceModelTrainingConfig(
            batch_size=512,
            epochs=20,
            learning_rate=0.0002,
            optimizer="adamw",
            warmup_epochs=5,
        ),
    },
    "single-digit-v1": {
        "model_class": SingleDigitModel,
        "model": SingleDigitModelConfig(
            encoder=ImageEncoderConfig(
                image_width=28,
                image_height=28,
                image_patch_width=7,
                image_patch_height=7,
                embedding_dimension=32,
                encoder_block_count=5,
                encoder_block=EncoderBlockConfig(
                    kq_dimension=16,
                    v_dimension=16,
                    embedding_dimension=32,
                    num_heads=2,
                    mlp_hidden_dimension=128,
                    mlp_dropout=0.2,
                ),
            )
        ),
        "model_trainer": SingleDigitModelTrainer,
        "training": SingleDigitModelTrainingConfig(
            batch_size=256,
            epochs=50,
            learning_rate=0.001,
            schedulers=["ReduceLROnPlateau"],
        ),
    },
}

DEFAULT_MODEL_NAME=list(DEFAULT_MODEL_PARAMETERS.keys())[0]

if __name__ == "__main__":
   for model_name, parameters in DEFAULT_MODEL_PARAMETERS.items():
        best_version = f"{model_name}-best"
        print(f"Loading Model: {best_version}")

        trainer = ModelTrainerBase.load_with_model(best_version)
        print(f"Latest validation metrics: {trainer.latest_validation_results}")
   
        print(f"Running model to check it's working...")
        trainer.run_validation()