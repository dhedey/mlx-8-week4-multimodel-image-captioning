# Run as uv run -m model.continue_train
import argparse
from .default_models import DEFAULT_MODEL_NAME
from .wandb_config import WANDB_PROJECT_NAME, WANDB_ENTITY
from .trainer import ModelTrainerBase, TrainingOverrides, ImageCaptioningModelTrainer
from .common import upload_model_artifact, ModelBase
import wandb
import os

trainer = ImageCaptioningModelTrainer.load_with_model("qwen-base-captioner-v1-best")

trainer.model.model_name = "qwen-base-captioner-v1-pirate"
trainer.config.dataset_kind = "pirate"
trainer.config.epochs = 3

trainer.save_model()