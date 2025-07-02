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
from typing import Optional, Self, Any
from peft import LoraConfig, TaskType, get_peft_model

from sympy.stats.rv import probability

from .common import ModelBase, ModuleConfig, TrainingConfig, Field
from .modules.encoder import EncoderBlockConfig
from .modules.decoder import DecoderBlockConfig, DecoderBlock

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def embedding_dimension(self) -> int:
        raise NotImplementedError("Should be implemented in a subclass.")

    def pre_process(self, images: list[PIL.Image.Image]) -> torch.Tensor:
        raise NotImplementedError("Should be implemented in a subclass.")

    def forward(self, preprocessed: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Should be implemented in a subclass.")

class ClipImageEncoderConfig(ModuleConfig):
    model_name: str = "openai/clip-vit-base-patch32"
    tokens_per_image: int
    model_embedding_dimension: int
    freeze_visual_model: bool = True

class ClipImageEncoder(ImageEncoder):
    def __init__(self, config: ClipImageEncoderConfig):
        super().__init__()
        self.processor = transformers.CLIPProcessor.from_pretrained(config.model_name, use_fast=True)
        self.model = transformers.CLIPModel.from_pretrained(config.model_name)
        self.model.text_model = None # Remove unneeded weights

        if config.freeze_visual_model:
            self.model.requires_grad_(False)
    
        self.config = config
        self.linear_mapping = nn.Linear(
            self.model.config.projection_dim,
            config.tokens_per_image * config.model_embedding_dimension,
        )

    def pre_process(self, images: list[PIL.Image.Image]) -> torch.Tensor:
        # Pre-processing resizes all images to the expected size
        # (batch_size, 3, height = 244, width = 244)
        return self.processor(images=images, return_tensors="pt")["pixel_values"]
    
    def forward(self, preprocessed: torch.Tensor) -> torch.Tensor:
        preprocessed = preprocessed.to(self.model.device)

        # (batch_size, model_projection_dim) => (batch_size, image_embedding_dimension)
        image_vector = self.model.get_image_features(pixel_values=preprocessed)
        # (batch_size, model_projection_dim) => (batch_size, tokens_per_image * model_embedding_dimension)
        outputs = self.linear_mapping(image_vector)
        # (batch_size, tokens_per_image, model_embedding_dimension)
        return einops.rearrange(
            outputs,
            "batch (tokens_per_image embedding_dimension) -> batch tokens_per_image embedding_dimension",
            embedding_dimension=self.config.model_embedding_dimension,
        )

class TextModel(nn.Module):
    def embedding_dimension(self) -> int:
        raise NotImplementedError("Should be implemented in a subclass.")

    def tokenize_to_fixed_length_with_start_and_separator(self, texts: list[str], fixed_length: int) -> torch.Tensor:
        raise NotImplementedError("Should be implemented in a subclass.")

    def token_ids_to_string(self, token_ids: list[int]) -> str:
        raise NotImplementedError("Should be implemented in a subclass.")

    def embed_for_model(self, preprocessed: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Should be implemented in a subclass.")

    def unembed(self, embedding: torch.Tensor) -> torch.Tensor:
        """Should take an embedding to logits over the token space"""
        raise NotImplementedError("Should be implemented in a subclass.")

    @property
    def separator_token_id(self) -> int:
        raise NotImplementedError("Should be implemented in a subclass.")

    @property
    def padding_token_id(self) -> int:
        raise NotImplementedError("Should be implemented in a subclass.")

    @property
    def image_start_token_id(self) -> int:
        raise NotImplementedError("Should be implemented in a subclass.")

    @property
    def caption_start_token_id(self) -> int:
        raise NotImplementedError("Should be implemented in a subclass.")


class BertTextModelConfig(ModuleConfig):
    model_embedding_dimension: int

class BertTextModel(TextModel):
    def __init__(self, config: BertTextModelConfig):
        super().__init__()


        # NOTE - We'll use [SEP] to mark the start and end of the caption, as per BERT's
        # pretrained convention.
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        prev_token_count = len(tokenizer)
        tokenizer.add_tokens(["[IMG_START]"])
        self._image_start_token_id = prev_token_count
        self.tokenizer = tokenizer

        bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        bert_model.resize_token_embeddings(len(tokenizer))  # Resize to accommodate new tokens

        self.bert_model = bert_model
        self.model_embedding = nn.Linear(
            self.bert_model.config.hidden_size,
            config.model_embedding_dimension,
        )
        self.model_unembedding = nn.Linear(
            config.model_embedding_dimension,
            len(tokenizer),
        )

    def tokenize_to_fixed_length_with_start_and_separator(self, texts: list[str], fixed_length: int) -> torch.Tensor:
        # By default this will add a [CLS] token at the start and a [SEP] token at the end
        return self.tokenizer(
            texts,
            max_length=fixed_length,
            return_tensors="pt",
            padding=transformers.utils.PaddingStrategy.MAX_LENGTH,
            truncation=True,
        )["input_ids"]

    def token_ids_to_string(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def embedding_dimension(self) -> int:
        return self.bert_model.config.hidden_size

    def embed_for_model(self, token_ids: torch.Tensor) -> torch.Tensor:
        temporarily_add_batch_dimension = token_ids.ndim == 1
        if temporarily_add_batch_dimension:
            # Add a batch dimension so that Bert doesn't error
            token_ids = token_ids.unsqueeze(0)

        attention_mask = token_ids != self.padding_token_id  # Elementwise

        bert_embedding = self.bert_model(
            input_ids=token_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        model_embeddings = self.model_embedding(bert_embedding)

        if temporarily_add_batch_dimension:
            # Remove batch dimension again
            return model_embeddings.squeeze(0)
        else:
            return model_embeddings

    def unembed(self, embedding: torch.Tensor) -> torch.Tensor:
        # Bert doesn't have unembeddings in the model, we'll have to learn our own
        return self.model_unembedding(embedding)

    @property
    def separator_token_id(self) -> int:
        return self.tokenizer.sep_token_id

    @property
    def padding_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def image_start_token_id(self) -> int:
        return self._image_start_token_id

    @property
    def caption_start_token_id(self) -> int:
        return self.tokenizer.cls_token_id

class DecoderLayersConfig(ModuleConfig):
    decoder_layers: int
    decoder_block_config: DecoderBlockConfig

class DecoderLayers(nn.Module):
    def __init__(self, config: DecoderLayersConfig):
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(config.decoder_block_config) for _ in range(config.decoder_layers)]
        )

    def forward(self, residual_stream) -> torch.Tensor:
        for decoder in self.decoder_blocks:
            residual_stream = decoder(residual_stream)
        return residual_stream


class QwenMultiModalModelConfig(ModuleConfig):
    freeze_visual_model: bool

@dataclass
class Section:
    batch_size: int

@dataclass
class CaptionSection(Section):
    section_token_ids: torch.LongTensor
    """The <|section_start|> <|caption|> ...text... <|section_end|> tokens"""

@dataclass
class ImageSection(Section):
    prepared_image: Any

@dataclass
class SectionResult:
    pass

@dataclass
class CaptionSectionResult(SectionResult):
    section_logits: torch.Tensor

@dataclass
class ImageSectionResult(SectionResult):
    pass

class QwenMultiModalModel(nn.Module):
    def __init__(self, config: QwenMultiModalModelConfig):
        super().__init__()
        model_name = "Qwen/Qwen3-0.6B-Base"

        self.config = config

        self.tokenizer: transformers.Qwen2TokenizerFast = transformers.AutoTokenizer.from_pretrained(model_name)
        self.auto_model: transformers.AutoModelForCausalLM = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.qwen_model: transformers.Qwen3Model = self.auto_model.model

        # Enable LORA on the Qwen model. get_peft_model actually changes it in-place
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=32,
            lora_dropout=0.05
        )
        self.peft_model = get_peft_model(
            self.auto_model,
            lora_config,
        )

        # SPECIAL TOKENS
        # <|im_start|>, <|im_end|>, <|user|>, <|assistant|>:
        # > Chat/Instruct models: Used and understood.
        # > Base model: Present in vocab, but not used for chat structure by default.
        # > <|endoftext|>: Used by both.

        # Let's use the following structure:
        # <|im_start|><|image|>   ... <|im_end|>
        # <|im_start|><|caption|> ... <|im_end|>

        # Add custom tokens
        self.tokenizer.add_tokens(["<|image|>", "<|caption|>"], special_tokens=True)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image|>")
        self.caption_token_id = self.tokenizer.convert_tokens_to_ids("<|caption|>")
        self.start_section_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.end_section_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.padding_token_id = self.tokenizer.pad_token_id
        self.qwen_model.resize_token_embeddings(len(self.tokenizer))

        self.embedding_dimension = self.qwen_model.config.hidden_size

        self.image_encoder = ClipImageEncoder(ClipImageEncoderConfig(
            tokens_per_image=1,  # CLIP encodes to a 512-dimensional vector already
            model_embedding_dimension=self.embedding_dimension,
            freeze_visual_model=self.config.freeze_visual_model,
        ))

        special_tokens = {
            "<|pad|>": self.padding_token_id,
            "<|section_start|>": self.start_section_token_id,
            "<|section_end|>": self.end_section_token_id,
            "<|image|>": self.image_token_id,
            "<|caption|>": self.caption_token_id,
        }
        print(f"Special token ids: {special_tokens}")

    def preprocess_images(self, images) -> ImageSection:
        return ImageSection(
            batch_size=len(images),
            prepared_image=self.image_encoder.pre_process(images)
        )

    def preprocess_captions(self, captions: list[str]) -> CaptionSection:
        batch_size = len(captions)

        # We want an end token before padding, but the tokenizer doesn't add it automatically.
        # So instead we get the tokenizer to just return lists and we put it into a tensor manually
        token_id_lists = self.tokenizer(
            captions,
            return_tensors=None,
            padding=transformers.utils.PaddingStrategy.DO_NOT_PAD,
        )["input_ids"]
        token_id_lists_max_length = max(len(ids) for ids in token_id_lists)
        section_length = token_id_lists_max_length + 3  # +3 for <|im_start|>, <|caption|>, <|im_end|>
        token_ids_tensor = torch.zeros((batch_size, section_length), dtype=torch.long)
        torch.fill(token_ids_tensor, self.padding_token_id)  # Fill with padding token id
        for i, token_ids in enumerate(token_id_lists):
            token_id_length = len(token_ids)
            token_ids_tensor[i, 0] = self.start_section_token_id
            token_ids_tensor[i, 1] = self.caption_token_id
            for j in range(token_id_length):
                token_ids_tensor[i, j + 2] = token_ids[j]
            token_ids_tensor[i, 2 + token_id_length] = self.end_section_token_id
            # We have already filled with padding ids

        return CaptionSection(
            batch_size=batch_size,
            section_token_ids=token_ids_tensor,
        )
    
    def token_ids_to_text(self, token_ids: list[int]) -> str:
        # Convert token ids to text using the tokenizer
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def embed_token_id(self, token_id: int, batch_size, device) -> torch.Tensor:
        input = torch.tensor([token_id], dtype=torch.long).to(device)
        embedding: torch.Tensor = self.qwen_model.embed_tokens(input)
        return einops.repeat(embedding, '1 embedding -> batch_size 1 embedding', batch_size=batch_size)

    def forward(self, sections: list[Section]) -> list[SectionResult]:
        # Go through each section, embed then, concatenate the embeddings, then run the model,
        # then return the results for each section.

        embedder = SectionEmbedder()
        device = next(self.parameters()).device
        for section in sections:
            embedder.start_section()
            match section:
                case ImageSection():
                    batch_size = section.batch_size
                    prepared_image = section.prepared_image
                    embedder.add(self.embed_token_id(self.start_section_token_id, batch_size, device))
                    embedder.add(self.embed_token_id(self.image_token_id, batch_size, device))
                    embedder.add(self.image_encoder(prepared_image))
                    embedder.add(self.embed_token_id(self.end_section_token_id, batch_size, device))
                case CaptionSection():
                    section_token_ids = section.section_token_ids
                    section_token_ids = section_token_ids.to(self.qwen_model.device)
                    embedder.add(self.qwen_model.embed_tokens(section_token_ids))
            embedder.end_section(section)

        # (Batch, Sequence, Embedding)
        final_hidden_state = self.qwen_model(
            inputs_embeds=torch.cat(embedder.embeddings, dim=-2)  # Concatenate along the sequence dimension,
        ).last_hidden_state

        section_results = []

        for section_data in embedder.section_offsets:
            start_offset = section_data["start_offset"]
            end_offset = section_data["end_offset"]
            match section:
                case ImageSection():
                    section_results.append(ImageSectionResult())
                case CaptionSection():
                    section_final_state = final_hidden_state[:, start_offset:end_offset, :]
                    section_results.append(CaptionSectionResult(
                        section_logits=self.auto_model.lm_head(section_final_state)
                    ))

        return section_results

class SectionEmbedder:
    def __init__(self):
        self.embeddings = []
        self.current_sequence_offset = 0
        self.section_offsets = []
        self.current_section_start_offset = 0

    def start_section(self):
        self.current_section_start_offset = self.current_sequence_offset

    def add(self, embeddings):
        self.embeddings.append(embeddings)
        self.current_sequence_offset += embeddings.shape[1]

    def end_section(self, section):
        self.section_offsets.append({
            "section": section,
            "start_offset": self.current_section_start_offset,
            "end_offset": self.current_sequence_offset,
        })

class ImageCaptioningModelV2Config(ModuleConfig):
    tokens_per_image: int
    freeze_image_weights: bool

class ImageCaptioningModelV2(ModelBase):
    def __init__(self, model_name: str, config: ImageCaptioningModelV2Config):
        super().__init__(model_name=model_name, config=config)
        self.config = config

        self.multi_modal_model: QwenMultiModalModel = QwenMultiModalModel(QwenMultiModalModelConfig(
            freeze_visual_model=self.config.freeze_image_weights,
        ))

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
    def padding_token_id(self):
        return self.multi_modal_model.padding_token_id

    def generate_caption(self, image) -> str:
        collated_batch = self.collate([{"image": image, "caption": ""}])
        output_token_ids = []

        max_generated_caption_length = 50 # Avoid possible infinite loops
        is_truncated = False

        for i in range(max_generated_caption_length):
            result: CaptionSectionResult = self.forward(collated_batch)
            caption_logits = result.section_logits
            next_token_logits = caption_logits[0, i + 1, :] # batch_index 0, sequence_index 1, take each token logit
            next_token_id = next_token_logits.argmax().item()
            # probabilities = next_token_logits.softmax(dim=-1)
            # next_token_id = torch.searchsorted(probabilities.cumsum(0), torch.rand(1)).item()
            if next_token_id == self.multi_modal_model.end_section_token_id:
                break
            else:
                output_token_ids.append(next_token_id)
                caption_section: CaptionSection = collated_batch["caption"]
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

class ImageCaptioningModelConfig(ModuleConfig):
    embedding_dimension: int
    max_tokens_per_caption: int
    tokens_per_image: int

    num_layers: int
    heads_per_layer: int
    attention_kq_dimension: int
    attention_v_dimension: int
    rope_enabled: bool

    mlp_hidden_dimension: int
    mlp_dropout: float

    freeze_image_weights: bool
    freeze_caption_weights: bool

class ImageCaptioningModel(ModelBase):
    def __init__(self, model_name: str, config: ImageCaptioningModelConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config

        self.image_encoder: ImageEncoder = ClipImageEncoder(ClipImageEncoderConfig(
            model_embedding_dimension = self.config.embedding_dimension,
            tokens_per_image = self.config.tokens_per_image,
            freeze_visual_model= self.config.freeze_image_weights,
        ))
        self.text_model: TextModel = BertTextModel(BertTextModelConfig(
            model_embedding_dimension = self.config.embedding_dimension,
        ))
        if self.config.freeze_caption_weights:
            self.text_model.requires_grad_(False)
        self.decoder = DecoderLayers(DecoderLayersConfig(
            decoder_layers = self.config.num_layers,
            decoder_block_config = DecoderBlockConfig(
                embedding_dimension = self.config.embedding_dimension,
                num_heads = self.config.heads_per_layer,
                kq_dimension = self.config.attention_kq_dimension,
                v_dimension = self.config.attention_v_dimension,
                rope_enabled = self.config.rope_enabled,

                mlp_hidden_dimension = self.config.mlp_hidden_dimension,
                mlp_dropout = self.config.mlp_dropout,
            ),
        ))
        self.max_context_length = (
            1   # [IMG_START]
            + self.config.tokens_per_image
            + 1 # [SEP]
            + 1 # [CAPTION_START]
            + self.config.max_tokens_per_caption
            # We don't add the SEP here because we don't need it as an input
        )

    def collate(self, dataset_batch) -> dict[str, torch.Tensor]:
        # Assume we've preprocessed the dataset to have a single caption and single image

        images = [item["image"] for item in dataset_batch]
        captions = [item["caption"] for item in dataset_batch]

        processed_images = self.image_encoder.pre_process(images)
        tokenized_captions = self.text_model.tokenize_to_fixed_length_with_start_and_separator(
            captions,
            fixed_length = 1 + self.config.max_tokens_per_caption + 1, # Add room for start and separator
        )

        input_caption_token_ids = tokenized_captions[:, 0:-1] # [Start] <tokens>
        output_caption_token_ids = tokenized_captions[:, 1:]  # <tokens> [SEP]

        return {
            "images": processed_images,
            "input_caption_token_ids": input_caption_token_ids,
            "output_caption_token_ids": output_caption_token_ids,
        }

    def forward(self, collated_batch) -> torch.Tensor:
        device = self.get_device()
        images = collated_batch["images"].to(device)
        input_caption_token_ids = collated_batch["input_caption_token_ids"].to(device)

        # TRAIN / INFERENCE FLOW
        # => Take the tokens [IMG_START], [PAD] * ImgLength, [IMG_END], [SEP], Padded caption
        # => Embed it with the text model
        # => Replace the image with the image embeddings
        # => Run the decoder on the combined embeddings
        # => Run the text model's unembed
        # => Output the predicted logits for each token

        embedded_caption = self.text_model.embed_for_model(input_caption_token_ids)
        embedded_image = self.image_encoder(images)

        # Prepare concatenated embedding
        batch_size = len(images)

        residual_stream = self.text_model.embed_for_model(
            torch.tensor([self.text_model.padding_token_id], dtype=torch.long).to(device)
        ).expand((batch_size, self.max_context_length, self.config.embedding_dimension)).clone()

        embedded_image_start = self.text_model.embed_for_model(
            torch.tensor([self.text_model.image_start_token_id], dtype=torch.long).to(device)
        )
        embedded_image_end = self.text_model.embed_for_model(
            torch.tensor([self.text_model.separator_token_id], dtype=torch.long).to(device)
        )

        residual_stream[:, 0:1] = embedded_image_start
        residual_stream[:, 1:1 + self.config.tokens_per_image] = embedded_image
        residual_stream[:, 1 + self.config.tokens_per_image: 1 + self.config.tokens_per_image + 1] = embedded_image_end

        caption_start_offset = 1 + self.config.tokens_per_image + 1
        assert caption_start_offset + self.config.max_tokens_per_caption + 1 == self.max_context_length
        residual_stream[:, caption_start_offset:] = embedded_caption

        # Run decoder
        output = self.decoder(residual_stream)

        # Return next token logits
        logits = self.text_model.unembed(output)

        return logits[:, caption_start_offset:]

    @property
    def padding_token_id(self):
        return self.text_model.padding_token_id

    def generate_caption(self, image) -> str:
        collated_batch = self.collate([{"image": image, "caption": ""}])
        output_token_ids = []

        for i in range(1 + self.config.max_tokens_per_caption): # [START] then tokens
            logits = self.forward(collated_batch)
            next_token_logits = logits[0, i, :] # batch_index 0, sequence_index 1, take each token logit
            next_token_id = next_token_logits.argmax().item()
            # probabilities = next_token_logits.softmax(dim=-1)
            # next_token_id = torch.searchsorted(probabilities.cumsum(0), torch.rand(1)).item()
            if next_token_id == self.text_model.separator_token_id:
                break
            else:
                output_token_ids.append(next_token_id)

        return self.text_model.token_ids_to_string(output_token_ids)

if __name__ == "__main__":
   print("Run default_models instead of this file")