from dataclasses import dataclass, field
import datasets
import einops
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import statistics
import random
import wandb
import math
import os
from typing import Optional
import time

from .common import TrainingState, TrainingOverrides, ModelTrainerBase, ModelBase, TrainingConfig, BatchResults, ValidationResults
from .models import CaptionSection, CaptionSectionResult, ImageCaptioningModel, ImageCaptioningModel
from .prepared_datasets import generate_image_caption_datasets, noop_collate


class ImageCaptioningModelTrainingConfig(TrainingConfig):
    dataset_kind: str # "standard" or "pirate"
    print_after_batches: int = 1 # Override default
    validation_max_print_examples: int = 5

class ImageCaptioningModelTrainer(ModelTrainerBase):
    def __init__(
            self,
            model: ImageCaptioningModel,
            config: ImageCaptioningModelTrainingConfig,
            overrides: Optional[TrainingOverrides] = None,
            continuation: Optional[TrainingState] = None,
        ):
        super().__init__(model=model, config=config, overrides=overrides, continuation=continuation)

        # These are already stored in the base class. But setting them again helps the IDE understand their type.
        self.model = model
        self.config = config

        print("Preparing datasets...")
        train_dataset, eval_dataset = generate_image_caption_datasets(config.dataset_kind)

        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(eval_dataset)}")

        self.train_loader = self.create_dataloader(train_dataset, self.config.batch_size, 2, model.collate)
        self.test_loader = self.create_dataloader(eval_dataset, self.config.batch_size, 2, model.collate)
        self.uncollated_validation_loader = self.create_dataloader(eval_dataset, 5, 0, noop_collate)

        print()

    def create_dataloader(self, dataset, batch_size, num_workers, collate_fn):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.model.get_device() == 'cuda', 
            collate_fn=collate_fn
        )

    @property
    def train_data_loader(self):
        return self.train_loader

    @property
    def validation_data_loader(self):
        return self.test_loader

    def process_batch(self, collated_batch) -> BatchResults:
        device = self.model.get_device()

        caption_section: CaptionSection = collated_batch["caption"]

        caption_section_ids = caption_section.section_token_ids.to(device)
        caption_section_result: CaptionSectionResult = self.model(collated_batch)
        caption_section_logits = caption_section_result.section_logits.to(device)

        # We offset expected and actual by one
        # We remove the useless prediction for the SectionEnd token
        expected_token_ids = caption_section_ids[:, 1:]
        actual_output_logits = caption_section_logits[:, :-1, :]

        # Sanity check that we're attempting to learn that that there should be a section end
        assert (expected_token_ids[0, :] == self.model.special_token_ids.section_end).any()

        criterion = nn.CrossEntropyLoss(ignore_index=self.model.special_token_ids.padding)
        loss = criterion(
            # The CrossEntropyLoss expects the second dimension to be the token id dimension
            einops.rearrange(actual_output_logits, "batch position token_id -> batch token_id position"),
            expected_token_ids,
        )

        total_non_padding_predictions = (expected_token_ids != self.model.special_token_ids.padding).sum().item()

        return BatchResults(
            total_loss=loss,
            num_samples=total_non_padding_predictions,
            intermediates={
                "expected_ids": expected_token_ids,
                "logits": actual_output_logits,
            }
        )
    
    def custom_validation(self) -> Optional[dict]:
        print_example_count = 0
        for raw_batch in self.uncollated_validation_loader:
            for item in raw_batch:
                if print_example_count >= self.config.validation_max_print_examples:
                    break
                print(f"===== Example {print_example_count + 1} =====")
                print(f"- Actual caption   : {item["caption"]}")
                print(f"- Generated caption: ", end="", flush=True)
                for token in self.model.generate_caption_streaming(item["image"]):
                    print(token, end="", flush=True)
                print()
                print()
                print_example_count += 1

        return None
