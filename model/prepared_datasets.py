import datasets
import os
import huggingface_hub

def noop_collate(batch):
    return batch

def flickr30k_is_train(item) -> bool:
    return item["split"] == "train"

def flickr30k_is_val(item) -> bool:
    return item["split"] != "train"

def flickr30k_take_first_caption(dataset_batch):
    captions: list[str] = []
    images = []
    for image, image_captions in zip(dataset_batch["image"], dataset_batch["caption"]):
        for caption in image_captions:
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
                data_dir=data_folder,
                split="test",
            )
        case "pirate":
            print("Need to login so that you can have access to the private dataset")
            print("Visit https://huggingface.co/settings/tokens to get a token")
            huggingface_hub.login()
            ds = datasets.load_dataset(
                "david-edey/flickr30k-pirate-captions",
                data_dir=data_folder,
                split="test",
                token=True,
            )
        case _:
            raise ValueError(f"Unknown dataset kind: {dataset_kind}")


    train_dataset = ds.filter(flickr30k_is_train)
    train_dataset = train_dataset.map(flickr30k_take_first_caption, batched=True, remove_columns=ds.column_names)
    eval_dataset = ds.filter(flickr30k_is_val)
    eval_dataset = eval_dataset.map(flickr30k_take_first_caption, batched=True, remove_columns=ds.column_names)

    return train_dataset, eval_dataset
