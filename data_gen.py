from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch, os
from qwen_vl_utils import process_vision_info

# Load the first 1000 images from flickr30k

data_folder = os.path.join(os.path.dirname(__file__), "model/datasets")
dataset = load_dataset("nlphuji/flickr30k", split="test[:1000]", data_dir=data_folder)

# Load Qwen2.5-VL-3B-Instruct model and processor
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "auto"
print(f"Using {device} for model map.")
model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)

def pirate_caption(example):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": example["image"],
                },
                {"type": "text", "text": "Provide a short english description in the style of a pirate of this image. No more than 100 words."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, temperature=0.5, do_sample=True)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0].split("assistant")[-1].strip()
    example["caption"] = caption
    print(f"Generated caption: {caption}")
    return example

# Use map to generate captions
pirate_dataset = dataset.map(
    pirate_caption,
    batched=False,  # batched=True is possible with some models, but Qwen2.5-VL-3B-Instruct may require single-image processing
    desc="Generating pirate captions"
)

# Save locally
pirate_dataset.save_to_disk("flickr30k_pirate_captions")

# Optionally, push to the Hub
# pirate_dataset.push_to_hub("your-username/flickr30k-pirate-captions")