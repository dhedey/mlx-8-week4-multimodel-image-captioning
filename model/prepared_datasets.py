import datasets
import os
import huggingface_hub

def noop_collate(batch):
    return batch

def flickr30k_is_train(item) -> bool:
    return item["split"] == "train"

def flickr30k_is_val(item) -> bool:
    return item["split"] != "train"

def flickr30k_take_first_caption_truncated_v2(dataset_batch, max_caption_length: int = 300):
    captions: list[str] = []
    images = []
    for image, image_captions in zip(dataset_batch["image"], dataset_batch["caption"]):
        for caption in image_captions:
            if len(caption) > max_caption_length:
                # Find the last '.' before position max_caption_length
                cut_pos = caption.rfind('.', 0, max_caption_length - 1) + 1
                if cut_pos == 0:
                    cut_pos = caption.rfind('!', 0, max_caption_length - 1) + 1
                if cut_pos == 0:
                    cut_pos = caption.rfind('?', 0, max_caption_length - 1) + 1
                if cut_pos == 0:
                    cut_pos = max_caption_length
                caption = caption[:cut_pos]

            captions.append(caption)
            images.append(image)
            break  # Only use the first caption for each image for now to speed up epochs

    return {
        "image": images,
        "caption": captions,
    }

def generate_image_caption_datasets(dataset_kind = "standard"):
    data_folder = os.path.join(os.path.dirname(__file__), "datasets")

    match dataset_kind:
        case "standard":
            # The dataset is improperly pre-split, and just has a train partition. Use that.
            ds = datasets.load_dataset(
                "nlphuji/flickr30k",
                cache_dir=data_folder,
                split="test",
            )
        case "pirate":
            print("You may need to login so that you can have access to the private dataset.")
            print("If so, visit https://huggingface.co/settings/tokens to get a token (and you may need to uncomment this line below)")
            print()
            # huggingface_hub.login()
            ds = datasets.load_dataset(
                "david-edey/flickr30k-pirate-captions",
                cache_dir=data_folder,
                token=True,
                split="test",
            )
        case _:
            raise ValueError(f"Unknown dataset kind: {dataset_kind}")


    train_dataset = ds.filter(flickr30k_is_train)
    train_dataset = train_dataset.map(flickr30k_take_first_caption_truncated_v2, batched=True, remove_columns=ds.column_names)
    eval_dataset = ds.filter(flickr30k_is_val)
    eval_dataset = eval_dataset.map(flickr30k_take_first_caption_truncated_v2, batched=True, remove_columns=ds.column_names)

    return train_dataset, eval_dataset
