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
from typing import Optional, Self, Any, Union, Generator
from peft import LoraConfig, TaskType, get_peft_model

from sympy.stats.rv import probability

from .modules.bert_custom_multi_modal_model import BertMultiModalModelConfig, BertMultiModalModel
from .modules.qwen_multi_modal_model import QwenMultiModalModelConfig, QwenMultiModalModel
from .modules.multi_modal_model import Section, CaptionSectionResult, CaptionSection, MultiModalModelResult, \
    SpecialTokenIds
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
        section_results: MultiModalModelResult = self.multi_modal_model([
            collated_batch["image"],
            collated_batch["caption"],
        ])
        result = section_results.sections[1]
        assert isinstance(result, CaptionSectionResult) # Make PyCharm happy
        return result

    @property
    def special_token_ids(self) -> SpecialTokenIds:
        return self.multi_modal_model.special_token_ids

    def generate_caption_streaming(self, image, max_token_length: int = 100) -> Generator[str, None, None]:
        collated_batch = self.collate([{"image": image, "caption": ""}])
        caption_section: CaptionSection = collated_batch["caption"]
        caption_section.section_token_ids = caption_section.section_token_ids[:, 0:2] # Start with <|im_start|> <|caption|>

        end_section_token_id = self.multi_modal_model.special_token_ids.section_end

        initial_result: MultiModalModelResult = self.multi_modal_model([
            collated_batch["image"],
            collated_batch["caption"],
        ])
        caption_section_result = initial_result.sections[1]
        assert isinstance(caption_section_result, CaptionSectionResult)
        caption_logits = caption_section_result.section_logits
        cache = initial_result.cache

        # Read the next token from the <|caption|> token
        next_token_logits = caption_logits[0, 1, :]
        next_token_id = next_token_logits.argmax().item()
        yield self.multi_modal_model.token_ids_to_text([next_token_id])

        for i in range(max_token_length):
            next_token_embed = self.multi_modal_model.embed_token_id(next_token_id, batch_size=1)
            single_token_logits, cache = self.multi_modal_model.continue_forward(next_token_embed, cache)
            next_token_logits = single_token_logits[0, 0, :]
            next_token_id = next_token_logits.argmax().item()
            # probabilities = next_token_logits.softmax(dim=-1)
            # next_token_id = torch.searchsorted(probabilities.cumsum(0), torch.rand(1)).item()
            if next_token_id == end_section_token_id:
                break
            else:
                yield self.multi_modal_model.token_ids_to_text([next_token_id])
        else:
            yield " [TRUNCATED]"

    def generate_caption(self, image, max_token_length: int = 100) -> str:
        full_caption = ""
        for token in self.generate_caption_streaming(image, max_token_length=max_token_length):
            full_caption += token
        return full_caption

if __name__ == "__main__":
   print("Run default_models instead of this file")