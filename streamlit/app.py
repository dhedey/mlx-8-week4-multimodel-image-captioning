import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import torch
import random
import io
import sys
import os
import time
from uuid import uuid4
import streamlit.components.v1 as components

# Add the project root to the Python path, so we can import the model modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import model.models as models

# Invert the display_names dictionary to map display names back to keys
display_names = {
    "qwen-base-captioner-v1-best": "CLIP + Qwen",
    "qwen-base-captioner-v1-pirate-best": "CLIP + Qwen + Pirate fine-tuning",
}
name_to_key = {v: k for k, v in display_names.items()}

# --- Model Selection ---
mode_col, model_select_col = st.columns([5, 5])

with mode_col:
    mode = st.selectbox(
        "Choose a mode:",
        options=["Upload", "Webcam"],
        index=0,
    )
with model_select_col:
    model_display_name = st.selectbox(
        "Choose a model to test:",
        options=list(name_to_key.keys()),
        index=0,
    )
model_key = name_to_key[model_display_name]

@st.cache_resource
def load_model(model_name) -> models.ImageCaptioningModel:
    """Loads and caches the selected model."""
    return models.ImageCaptioningModel.load_for_evaluation(model_name)

image_col, generate_col = st.columns([5, 5])

with image_col:
    image = None
    match mode:
        case "Upload":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    image = None
        case "Webcam":
            camera_frame = st.camera_input("Capture an image", key="webcam_input")
            if camera_frame is not None:
                try:
                    image = Image.open(camera_frame).convert("RGB")
                except Exception as e:
                    st.error(f"Error loading webcam image: {e}")
                    image = None        

with generate_col:
    model = load_model(model_key)
    if image is None:
        st.info("Please upload an image")
    else:
        caption = model.generate_caption_streaming(image, max_token_length=200)
        st.write_stream(caption)
