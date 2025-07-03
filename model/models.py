from dataclasses import dataclass
import PIL.Image
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
import PIL
from typing import Optional, Self, Any, Union
from peft import LoraConfig, TaskType, get_peft_model

from sympy.stats.rv import probability

from .modules.bert_custom_multi_modal_model import BertMultiModalModelConfig, BertMultiModalModel
from .modules.qwen_multi_modal_model import QwenMultiModalModelConfig, QwenMultiModalModel
from .modules.multi_modal_model import Section, CaptionSectionResult, CaptionSection
from .common import ModelBase, ModuleConfig, TrainingConfig, Field

class ImageCaptioningModelConfig(ModuleConfig):
    model: Union[QwenMultiModalModelConfig, BertMultiModalModelConfig]

class ImageCaptioningModel(ModelBase):
    def __init__(self, model_name: str, config: ImageCaptioningModelConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config

        match config.model:
            case QwenMultiModalModelConfig():
                self.multi_modal_model = QwenMultiModalModel(config.model)
            case BertMultiModalModelConfig():
                self.multi_modal_model = BertMultiModalModel(config.model)
            case _:
                raise ValueError(f"Unknown model config type: {config.model.__class__.__name__}")

    def collate(self, dataset_batch) -> dict[str, Section]:
        # Assume we've preprocessed the dataset to have a single caption and single image

        images = [item["image"] for item in dataset_batch]
        image_section = self.multi_modal_model.preprocess_images(images)

        captions = [item["caption"] for item in dataset_batch]
        caption_section = self.multi_modal_model.preprocess_captions(captions)

        return {
            "image": image_section,
            "caption": caption_section,
        }

    def forward(self, collated_batch) -> CaptionSectionResult:
        section_results = self.multi_modal_model([
            collated_batch["image"],
            collated_batch["caption"],
        ])
        return section_results[1]

    @property
    def padding_token_id(self) -> int:
        return self.multi_modal_model.special_token_ids.padding

    def generate_caption(self, image, max_token_length: int = 100) -> str:
        collated_batch = self.collate([{"image": image, "caption": ""}])
        caption_section: CaptionSection = collated_batch["caption"]
        caption_section.section_token_ids = caption_section.section_token_ids[:, 0:2] # Start with <|im_start|> <|caption|>
        output_token_ids = []

        max_token_length = 50 # Avoid possible infinite loops
        is_truncated = False
        end_section_token_id = self.multi_modal_model.special_token_ids.section_end

        for i in range(max_token_length):
            result: CaptionSectionResult = self.forward(collated_batch)
            caption_logits = result.section_logits
            next_token_logits = caption_logits[0, i + 1, :] # batch_index 0, sequence_index 1, take each token logit
            next_token_id = next_token_logits.argmax().item()
            # probabilities = next_token_logits.softmax(dim=-1)
            # next_token_id = torch.searchsorted(probabilities.cumsum(0), torch.rand(1)).item()
            if next_token_id == end_section_token_id:
                break
            else:
                output_token_ids.append(next_token_id)
                caption_section = collated_batch["caption"]
                assert isinstance(caption_section, CaptionSection) # Fix pycharm complaining

                caption_section.section_token_ids = torch.cat([
                    caption_section.section_token_ids,
                    torch.tensor([[next_token_id]], dtype=torch.long, device=caption_section.section_token_ids.device),
                ], dim=1)
        else:
            is_truncated = True

        output = self.multi_modal_model.token_ids_to_text(output_token_ids)
        if is_truncated:
            output += " [TRUNCATED]"

        return output

if __name__ == "__main__":
   print("Run default_models instead of this file")