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
from .models import ImageCaptioningModel, ImageCaptioningModelConfig
from .trainer import ModelTrainerBase, ImageCaptioningTrainer, ImageCaptioningTrainingConfig
from .wandb_config import WANDB_ENTITY, WANDB_PROJECT_NAME

DEFAULT_MODEL_PARAMETERS = {
    "image-captioner-v1": {
        "model_class": ImageCaptioningModel,
        "model": ImageCaptioningModelConfig(
            embedding_dimension = 256,
            max_tokens_per_caption = 80,
            tokens_per_image = 1, # CLIP encoder encodes to a 512 dimensional vector already

            num_layers = 8,
            heads_per_layer = 8,
            attention_kq_dimension = 128,
            attention_v_dimension = 128,

            mlp_hidden_dimension = 256 * 4,
            mlp_dropout = 0.2,

            freeze_caption_weights = True,
            freeze_image_weights = True,
        ),
        "model_trainer": ImageCaptioningTrainer,
        "training": ImageCaptioningTrainingConfig(
            batch_size=128,
            epochs=20,
            learning_rate=0.001,
            optimizer="adamw",
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