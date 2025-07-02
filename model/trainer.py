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

from .common import TrainingState, TrainerOverrides, ModelTrainerBase, ModelBase, TrainingConfig, BatchResults, ValidationResults
from .composite_dataset import CompositeDataset, DavidCompositeDataset, sequence_collate_fn, BesCombine
from .models import CaptionSection, CaptionSectionResult, ImageCaptioningModel, ImageCaptioningModelV2

def noop_collate(batch):
    return batch


class ImageCaptioningV2TrainingConfig(TrainingConfig):
    print_after_batches: int = 1
    epoch_max_batches: Optional[int] = None
    validation_max_batches: Optional[int] = None
    validation_max_print_examples: int = 5

class ImageCaptioningV2Trainer(ModelTrainerBase):
    def __init__(
            self,
            model: ImageCaptioningModelV2,
            config: ImageCaptioningV2TrainingConfig,
            overrides: Optional[TrainerOverrides] = None,
            continuation: Optional[TrainingState] = None,
        ):
        super().__init__(model=model, config=config, overrides=overrides, continuation=continuation)

        # These are already stored in the base class. But setting them again helps the IDE understand their type.
        self.model = model
        self.config = config

        print("Preparing datasets...")
        data_folder = os.path.join(os.path.dirname(__file__), "datasets")

        # The dataset is improperly pre-split, and just has a train partition. Use that.
        ds = datasets.load_dataset("nlphuji/flickr30k", data_dir=data_folder)["test"]

        def dataset_flatten(dataset_batch):
            captions: list[str] = []
            images = []
            for image, image_captions in zip(dataset_batch["image"], dataset_batch["caption"]):
                for caption in image_captions:
                    captions.append(caption)
                    images.append(image)

            return {
                "image": images,
                "caption": captions,
            }

        train_dataset = ds.filter(lambda item: item["split"] == "train").map(dataset_flatten, batched=True, remove_columns=ds.column_names)
        eval_dataset = ds.filter(lambda item: item["split"] != "train").map(dataset_flatten, batched=True, remove_columns=ds.column_names)

        device = self.model.get_device()
        pin_memory = device == 'cuda'
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory,
            collate_fn=model.collate
        )
        self.test_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory, # Speed up CUDA
            collate_fn=model.collate
        )
        self.uncollated_validation_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=5,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory,  # Speed up CUDA
            collate_fn=noop_collate
        )
        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(eval_dataset)}")
        print()

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
        total_loss = 0.0
        num_samples = 0

        for batch_idx, collated_batch in enumerate(self.test_loader):
            results = self.process_batch(collated_batch)

            total_loss += results.total_loss.item()
            num_samples += results.num_samples

            batch_num = batch_idx + 1
            if self.config.validation_max_batches is not None and batch_num >= self.config.validation_max_batches:
                print("Ending validation early due to self.config.validation_max_batches")
                break

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

        average_loss = total_loss / num_samples if num_samples > 0 else 0.0

        print(f"Validation complete: {num_samples} samples, {average_loss:.3g} average loss")
        print()

        return ValidationResults(
            epoch=self.epoch,
            average_training_loss=average_loss,
            validation_loss=average_loss,
        )


class ImageCaptioningTrainingConfig(TrainingConfig):
    print_after_batches: int = 1
    epoch_max_batches: Optional[int] = None
    validation_max_batches: Optional[int] = None
    validation_max_print_examples: int = 5

class ImageCaptioningTrainer(ModelTrainerBase):
    def __init__(
            self,
            model: ImageCaptioningModel,
            config: ImageCaptioningTrainingConfig,
            overrides: Optional[TrainerOverrides] = None,
            continuation: Optional[TrainingState] = None,
        ):
        super().__init__(model=model, config=config, overrides=overrides, continuation=continuation)

        # These are already stored in the base class. But setting them again helps the IDE understand their type.
        self.model = model
        self.config = config

        print("Preparing datasets...")
        data_folder = os.path.join(os.path.dirname(__file__), "datasets")

        # The dataset is improperly pre-split, and just has a train partition. Use that.
        ds = datasets.load_dataset("nlphuji/flickr30k", data_dir=data_folder)["test"]

        def dataset_flatten(dataset_batch):
            captions: list[str] = []
            images = []
            for image, image_captions in zip(dataset_batch["image"], dataset_batch["caption"]):
                for caption in image_captions:
                    captions.append(caption)
                    images.append(image)

            return {
                "image": images,
                "caption": captions,
            }

        train_dataset = ds.filter(lambda item: item["split"] == "train").map(dataset_flatten, batched=True, remove_columns=ds.column_names)
        eval_dataset = ds.filter(lambda item: item["split"] != "train").map(dataset_flatten, batched=True, remove_columns=ds.column_names)

        device = self.model.get_device()
        pin_memory = device == 'cuda'
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory,
            collate_fn=model.collate
        )
        self.test_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory, # Speed up CUDA
            collate_fn=model.collate
        )
        self.uncollated_validation_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=5,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory,  # Speed up CUDA
            collate_fn=noop_collate
        )
        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(eval_dataset)}")
        print()

    def get_train_data_loader(self):
        return self.train_loader

    def process_batch(self, collated_batch) -> BatchResults:
        caption_logits = self.model(collated_batch)
        caption_token_ids = collated_batch["output_caption_token_ids"].to(self.model.get_device())

        # The CrossEntropyLoss expects the second dimension to be the token id dimension
        reorganized_logits = einops.rearrange(
            caption_logits,
            "batch position token_id -> batch token_id position"
        )
        criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_token_id)
        loss = criterion(reorganized_logits, caption_token_ids)

        return BatchResults(
            total_loss=loss,
            num_samples=len(caption_token_ids),
            intermediates={
                "logits": caption_logits,
                "labels": caption_token_ids,
            }
        )
    
    def _validate(self) -> ValidationResults:
        total_loss = 0.0
        num_samples = 0

        for batch_idx, collated_batch in enumerate(self.test_loader):
            results = self.process_batch(collated_batch)

            total_loss += results.total_loss.item()
            num_samples += results.num_samples

            batch_num = batch_idx + 1
            if self.config.validation_max_batches is not None and batch_num >= self.config.validation_max_batches:
                print("Ending validation early due to self.config.validation_max_batches")
                break

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

        average_loss = total_loss / num_samples if num_samples > 0 else 0.0

        print(f"Validation complete: {num_samples} samples, {average_loss:.3g} average loss")
        print()

        return ValidationResults(
            epoch=self.epoch,
            average_training_loss=average_loss,
            validation_loss=average_loss,
        )
