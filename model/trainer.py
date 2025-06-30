from dataclasses import dataclass, field
import datasets
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
from .models import SingleDigitModel, DigitSequenceModel

class SingleDigitModelTrainingConfig(TrainingConfig):
    pass

class SingleDigitModelTrainer(ModelTrainerBase):
    def __init__(
            self,
            model: SingleDigitModel,
            config: SingleDigitModelTrainingConfig,
            overrides: Optional[TrainerOverrides] = None,
            continuation: Optional[TrainingState] = None,
        ):
        super().__init__(model=model, config=config, overrides=overrides, continuation=continuation)

        # These are already stored in the base class. But setting them again helps the IDE understand their type.
        self.model = model
        self.config = config

        print("Preparing datasets...")

        training_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True), # Scale to [0, 1]
            v2.RandomResize(28, 40),
            v2.RandomRotation(30),
            v2.RandomResizedCrop(size = 28, scale = (28.0/40, 28.0/40)),
        ])

        test_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True), # Scale to [0, 1]
        ])

        data_folder = os.path.join(os.path.dirname(__file__), "datasets")
        train_set = torchvision.datasets.MNIST(
            data_folder,
            download=True,
            transform=training_transform,
            train=True,
        )
        test_set = torchvision.datasets.MNIST(
            data_folder,
            download=True,
            transform=test_transform,
            train=False,
        )
        device = self.model.get_device()
        pin_memory = device == 'cuda'
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory, # Speed up CUDA
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory, # Speed up CUDA
        )
        print(f"Training set size: {len(train_set)}")
        print(f"Test set size: {len(test_set)}")
        print()

    def get_train_data_loader(self):
        return self.train_loader

    def process_batch(self, raw_batch) -> BatchResults:
        inputs, labels = raw_batch
        device = self.model.get_device()
        inputs = inputs.to(device)
        labels = labels.to(device)

        criterion = nn.CrossEntropyLoss()
        logits = self.model(inputs)
        loss = criterion(logits, labels)

        return BatchResults(
            total_loss=loss,
            num_samples=len(inputs),
            intermediates={
                "logits": logits,
                "labels": labels,
            }
        )
    
    def _validate(self) -> ValidationResults:
        total_loss = 0.0
        num_samples = 0

        totals_by_label = [0] * 10
        correct_by_label = [0] * 10

        total_correct = 0
        total_probability_of_correct = 0.0

        for raw_batch in self.test_loader:
            results = self.process_batch(raw_batch)

            logits = results.intermediates["logits"]
            labels = results.intermediates["labels"]
            probabilities = F.softmax(logits, dim=1)

            for instance_logits, instance_label, instance_probabilities in zip(logits, labels, probabilities):
                instance_label = instance_label.item()
                predicted_label = instance_logits.argmax().item()
                is_correct = predicted_label == instance_label
                totals_by_label[instance_label] += 1
                if is_correct:
                    total_correct += 1
                    correct_by_label[instance_label] += 1
                total_probability_of_correct += instance_probabilities[instance_label].item()

            total_loss += results.total_loss.item()
            num_samples += results.num_samples

        proportion_correct = total_correct / num_samples if num_samples > 0 else 0.0
        average_probability_of_correct = total_probability_of_correct / num_samples if num_samples > 0 else 0.0
        average_loss = total_loss / num_samples if num_samples > 0 else 0.0

        print(f"Validation complete: {num_samples} samples, {total_correct} correct, {average_loss:.3g} average loss")
        print(f"* Accuracy: {proportion_correct:.2%} correct")
        print(f"* Average Loss: {average_loss}")
        print(f"* Average confidence in correct answer: {average_probability_of_correct:.2%}")
        print()
        print("Proportion correct by actual label:")
        for i in range(10):
            label_prop_correct = correct_by_label[i] / totals_by_label[i] if totals_by_label[i] > 0 else 0
            print(f"* {i}: {label_prop_correct:.2%} ({correct_by_label[i]} of {totals_by_label[i]})")
        print()

        return ValidationResults(
            epoch=self.epoch,
            average_training_loss=average_loss,
            validation_loss=average_loss,
            proportion_correct=proportion_correct,
            average_probability_of_correct=average_probability_of_correct,
            totals_by_label=totals_by_label,
        )

class DigitSequenceModelTrainingConfig(TrainingConfig):
    training_set_size: int = 60000
    validation_set_size: int = 10000
    probability_of_skip: float = 0.3
    generator_kind: str = "bes"

class DigitSequenceModelTrainer(ModelTrainerBase):
    def __init__(
            self,
            model: DigitSequenceModel,
            config: DigitSequenceModelTrainingConfig,
            overrides: Optional[TrainerOverrides] = None,
            continuation: Optional[TrainingState] = None,
        ):
        super().__init__(model=model, config=config, overrides=overrides, continuation=continuation)

        # These are already stored in the base class.
        # But setting them again here helps the IDE understand their types.
        self.model = model
        self.config = config

        print("Preparing datasets...")

        device = self.model.get_device()

        pad_token_id = -1
        start_token_id = 10
        stop_token_id = 10

        def create_dataloader(kind: str, train: bool, size: int):
            pin_memory = device == 'cuda'
            match kind:
                case "bes":
                    dataset = BesCombine(
                        train=train,
                        h_patches=self.model.config.encoder.image_height // 28,
                        w_patches=self.model.config.encoder.image_width // 28,
                        length=size,
                        p_skip=self.config.probability_of_skip,
                        max_sequence_length=self.model.config.max_sequence_length,
                        pad_token_id=pad_token_id,
                        start_token_id=start_token_id,
                        stop_token_id=stop_token_id,
                    )
                    collate_fn = dataset.collate_fn
                case "nick":
                    dataset = CompositeDataset(
                        train=train,
                        canvas_size=(self.model.config.encoder.image_height, self.model.config.encoder.image_width),
                        digit_size=28,
                        length=size,
                        max_digits=self.model.config.max_sequence_length - 1,
                        pad_token_id=pad_token_id,
                        start_token_id=start_token_id,
                        stop_token_id=stop_token_id,
                    )
                    collate_fn = dataset.collate_fn
                case "david":
                    dataset = DavidCompositeDataset(
                        train=train,
                        length=size,
                        output_width=self.model.config.encoder.image_width,
                        output_height=self.model.config.encoder.image_height,
                        padding_token_id=pad_token_id,
                        start_token_id=start_token_id,
                        end_token_id=stop_token_id,
                        max_sequence_length=self.model.config.max_sequence_length,
                    )
                    collate_fn = dataset.collate_fn
                case "david-v2":
                    assert self.model.config.encoder.image_width == 70
                    assert self.model.config.encoder.image_height == 70
                    assert self.model.config.max_sequence_length == 10
                    dataset = DavidCompositeDataset(
                        train=train,
                        length=size,
                        use_cache=overrides.use_dataset_cache,
                        output_height=70,
                        output_width=70,
                        line_height_min=16,
                        line_height_max=30,
                        line_spacing_min=2,
                        line_spacing_max=20,
                        horizontal_padding_min=2,
                        horizontal_padding_max=60,
                        left_margin_offset=0,
                        first_line_offset=0,
                        image_scaling_min=0.8,
                        max_sequence_length=10,
                        padding_token_id=pad_token_id,
                        start_token_id=start_token_id,
                        end_token_id=stop_token_id,
                    )
                    collate_fn = dataset.collate_fn
                case _:
                    raise ValueError(f"Unknown dataset kind: {kind}")
                
            return size, torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=train,
                num_workers=2,
                collate_fn=collate_fn,
                pin_memory=pin_memory, # Speed up CUDA
            )

        generator_kind = self.config.generator_kind
        print("Using generator kind:", generator_kind)

        train_size, self.train_loader = create_dataloader(kind=generator_kind, train=True, size=self.config.training_set_size)
        test_size, self.test_loader = create_dataloader(kind=generator_kind, train=False, size=self.config.validation_set_size)

        print(f"Training set size: {train_size}")
        print(f"Test set size: {test_size}")

    def get_train_data_loader(self):
        return self.train_loader

    def process_batch(self, raw_batch) -> BatchResults:
        images, input_sequences, expected_sequences = raw_batch

        device = self.model.get_device()
        images = images.to(device)
        input_sequences = input_sequences.to(device)
        expected_sequences = expected_sequences.to(device) # Shape: (BatchSize, SequenceLength) => VocabularyIndex

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        logits: torch.Tensor = self.model(images, input_sequences)   # Shape: (BatchSize, SequenceLength, VocabularySize=11) => Probability Source

        # CrossEntropyLoss needs:
        # - Logits shaped like   (BatchSize, VocabularySize, ...Other Dimensions...)
        # - Expected shaped like (BatchSize, ...Other Dimensions...)
        loss = criterion(logits.transpose(-1, -2), expected_sequences.to(torch.long))

        return BatchResults(
            total_loss=loss,
            num_samples=len(images),
            intermediates={
                "logits": logits,
                "expected_sequences": expected_sequences,
            },
        )
    
    def _validate(self) -> ValidationResults:
        total_loss = 0.0

        totals_by_label = [0] * 11
        correct_by_label = [0] * 11

        total_subimages = 0
        total_subimages_correct = 0
        total_probability_of_subimage_correct = 0.0

        total_composites = 0
        correct_composites = 0

        for raw_batch in self.test_loader:
            batch_results = self.process_batch(raw_batch)

            logits = batch_results.intermediates["logits"]
            expected_sequences = batch_results.intermediates["expected_sequences"]
            loss = batch_results.total_loss
            probabilities = F.softmax(logits, dim=1)

            for instance_logits, instance_expected, instance_probabilities in zip(logits, expected_sequences, probabilities):
                all_correct = True
                for sub_image_logits, expected_label, sub_image_probabilities in zip(instance_logits, instance_expected, instance_probabilities):
                    if expected_label == -1:
                        continue
                    expected_label = expected_label.item()
                    predicted_label = sub_image_logits.argmax().item()
                    is_correct = predicted_label == expected_label
                    totals_by_label[expected_label] += 1
                    if is_correct:
                        total_subimages_correct += 1
                        correct_by_label[expected_label] += 1
                    else:
                        all_correct = False
                    
                    total_probability_of_subimage_correct += sub_image_probabilities[expected_label].item()
                    total_subimages += 1

                total_composites += 1
                if all_correct:
                    correct_composites += 1
            
            total_loss += loss.item()

        proportion_subimages_correct = total_subimages_correct / total_subimages if total_subimages > 0 else 0.0
        average_probability_of_subimages_correct = total_probability_of_subimage_correct / total_subimages if total_subimages > 0 else 0.0
        average_loss_per_subimage = total_loss / total_subimages if total_subimages > 0 else 0.0
        average_loss = total_loss / total_composites if total_composites > 0 else 0.0
        proportion_composites_correct = correct_composites / total_composites if total_composites > 0 else 0.0

        print(f"Validation complete: {total_composites} composites, containing {total_subimages} subimages, {total_subimages_correct} correct, {average_loss:.3g} average loss per composite")
        print(f"* Accuracy: {proportion_subimages_correct:.2%} subimages correct")
        print(f"* Accuracy: {proportion_composites_correct:.2%} composites fully correct")
        print(f"* Average loss for each subimage: {average_loss_per_subimage}")
        print(f"* Average confidence in each subimage: {average_probability_of_subimages_correct:.2%}")
        print()
        print("Proportion subimages correct by actual label:")
        for i in range(11):
            label_prop_correct = correct_by_label[i] / totals_by_label[i] if totals_by_label[i] > 0 else 0
            if i == 10:
                index_label = "END"
            else:
                index_label = f"[{i}]"
            print(f"* {index_label}: {label_prop_correct:.2%} ({correct_by_label[i]} of {totals_by_label[i]})")
        print()

        return ValidationResults(
            epoch=self.epoch,
            average_training_loss=average_loss,
            validation_loss=average_loss,
            average_loss_per_subimage=average_loss_per_subimage,
            proportion_subimages_correct=proportion_subimages_correct,
            proportion_composites_correct=proportion_composites_correct,
            average_probability_of_subimages_correct=average_probability_of_subimages_correct,
            totals_by_label=totals_by_label,
        )

# Back compatiblity for renamed model trainers
ModelTrainerBase.registered_types["ImageSequenceTransformerTrainer"] = DigitSequenceModelTrainer
ModelTrainerBase.registered_types["EncoderOnlyModelTrainer"] = SingleDigitModelTrainer
