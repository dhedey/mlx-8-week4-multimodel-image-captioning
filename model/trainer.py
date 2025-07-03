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

    def get_train_data_loader(self):
        return self.train_loader

    def process_batch(self, collated_batch) -> BatchResults:
        device = self.model.get_device()

        caption_section: CaptionSection = collated_batch["caption"]

        caption_section_ids = caption_section.section_token_ids.to(device)
        caption_section_result: CaptionSectionResult = self.model(collated_batch)
        caption_section_logits = caption_section_result.section_logits.to(device)

        expected_token_ids = caption_section_ids[:, 1:]           # Offset by one
        actual_output_logits = caption_section_logits[:, :-1, :]  # Remove the last SectionEnd token, so it aligns with the expected length

        criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_token_id)
        loss = criterion(
            # The CrossEntropyLoss expects the second dimension to be the token id dimension
            einops.rearrange(
                caption_section_logits[:, :-1, :],
                "batch position token_id -> batch token_id position"
            ),
            expected_token_ids,
        )

        total_non_padding_predictions = (expected_token_ids != self.model.padding_token_id).sum().item()

        return BatchResults(
            total_loss=loss,
            num_samples=total_non_padding_predictions,
            intermediates={
                "expected_ids": expected_token_ids,
                "logits": actual_output_logits,
            }
        )
    
    def _validate(self) -> ValidationResults:
        print_example_count = 0
        for raw_batch in self.uncollated_validation_loader:
            for item in raw_batch:
                if print_example_count >= self.config.validation_max_print_examples:
                    break
                print(f"Example {print_example_count + 1}")
                print(f"- Actual caption   : {item["caption"]}")
                caption = self.model.generate_caption(item["image"])
                print(f"- Generated caption: {caption}")
                print_example_count += 1

        total_loss = 0.0
        num_samples = 0

        for batch_idx, collated_batch in enumerate(self.test_loader):
            results = self.process_batch(collated_batch)

            total_loss += results.total_loss.item()
            num_samples += results.num_samples

            batch_num = batch_idx + 1
            if self.config.batch_limit is not None and batch_num >= self.config.batch_limit:
                print("Ending validation early due to self.config.batch_limit")
                break

        average_loss = total_loss / num_samples if num_samples > 0 else 0.0

        print(f"Validation complete: {num_samples} samples, {average_loss:.3g} average loss")
        print()

        return ValidationResults(
            epoch=self.epoch,
            average_training_loss=average_loss,
            validation_loss=average_loss,
        )
