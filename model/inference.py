from .common import select_device, ModelBase, ModuleConfig
from .models import DigitSequenceModel
from .default_models import DEFAULT_MODEL_PARAMETERS
from .composite_dataset import BesCombine, CompositeDataset, DavidCompositeDataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from pydantic import ValidationError
from typing import Self
import torchvision
import os
import torchvision.transforms.v2 as v2

def select_model(choice: str):
    match choice:
        case "best2by2":
            model_name = "multi-digit-v1"
            model_path = ModelBase.model_path("best2by2")
        case "best4by4_var":
            model_name = "multi-digit-v1"
            model_path = ModelBase.model_path("best4by4")
        case "best5by5_scramble":
            model_name = "multi-digit-v1"
            model_path = ModelBase.model_path("multi-digit-scrambled-best")
        case "scrambled-v2":
            model_name = "multi-digit-scrambled-v2"
            model_path = ModelBase.model_path("multi-digit-scrambled-v2-best")
        case _:
            raise ValueError(f"Invalid model choice: {choice}")

    return {
        "model_name": model_name,
        "model_path": model_path,
    }

def get_data_info(choice : str):
    match choice:
        case "best2by2":
            return {
                "h_patches": 2,
                "w_patches": 2,
                "image_size": (56, 56),
                "variable_length": False,
                "type": "bes",
            }
        case "best4by4_var":
            return {
                "h_patches": 4,
                "w_patches": 4,
                "image_size": (112, 112),
                "variable_length": True,
                "type": "bes",
            }
        case "best5by5_scramble":
            return {
                "h_patches": 5,
                "w_patches": 5,
                "image_size": (140, 140),
                "variable_length": True,
                "type": "david",
            }
        case "scrambled-v2":
            return {
                "image_size": (70, 70),
                "variable_length": True,
                "type": "david-v2",
            }
        case _:
            raise ValueError(f"Invalid model choice: {choice}")

def predict_sequence(model, image) -> np.ndarray:
    #start with only the start token
    seq = torch.tensor([10] + [-1] * (model.config.max_sequence_length))
    for i in range(model.config.max_sequence_length):
        logits = model(image, seq[:-1])[0]
        #get the logits for the next token
        next_token_logits = logits[i, :]
        #sample the next token
        next_token = torch.argmax(next_token_logits)
        #add the next token to the sequence
        seq[i + 1] = next_token
        #stop if the next token is the end token
        if next_token == 10:
            break
    return seq[seq != -1][1:].cpu().numpy()

def display_test_images(model, data_info):
    h_patches = data_info["h_patches"]
    w_patches = data_info["w_patches"]
    p_skip = 0.2 if data_info["variable_length"] else 0
    if data_info["type"] == "bes":
        test_ds = BesCombine(train=False, h_patches=h_patches, w_patches=w_patches, length=100, p_skip=p_skip)
    elif data_info["type"] == "nick":
        test_ds = CompositeDataset(train=False, length=100, canvas_size=(28 * h_patches, 28 * w_patches), min_digits=h_patches * w_patches, max_digits=h_patches * w_patches)
    elif data_info["type"] == "david":
        test_ds = DavidCompositeDataset(
            train=False,
            length=10,
            output_width=model.config.encoder.image_width,
            output_height=model.config.encoder.image_height,
            max_sequence_length=model.config.max_sequence_length,
            padding_token_id = -1,
            start_token_id = 10,
            end_token_id = 10,
        )
    else:
        raise ValueError(f"Invalid data type: {data_info['type']}")

    for item in test_ds:
        test_image = item[0]
        while len(test_image.shape) < 3:
            test_image = test_image.unsqueeze(0) # Add channel dimension
        out_seq = predict_sequence(model, test_image)
        seq_str = "".join((str(int(x)) if x<10 else "<END>") for x in out_seq)
        plt.imshow(test_image.squeeze(0).cpu().numpy(), cmap="gray")
        plt.title(f"Predicted sequence: {seq_str}")
        plt.show()
        yield test_image, out_seq

if __name__ == "__main__":
    device = select_device()

    # Load the model and data info   
    name = "best4by4_var"
    all_names = ["best5by5_scramble", "best4by4_var", "best2by2"]
    display_names = {
        "best5by5_scramble": "5 x 5 scrambled",
        "best4by4_var": "4 x 4 on-grid",
        "best2by2": "2 x 2 on-grid (fixed length)",
    }
    model_path = select_model(name)["model_path"]
    data_info = get_data_info(name)

    model = DigitSequenceModel.load_for_evaluation(model_path=model_path, device=device)

    for test_image, out_seq in display_test_images(model, data_info):
        print("Generating another image... Press Ctrl+C to stop, or close image to get another.")
