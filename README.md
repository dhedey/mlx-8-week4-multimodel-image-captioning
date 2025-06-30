# Machine Learning Institute - Week 4 - Multimodal architectures / Image Captioning

This week, we are experimenting with multi-model transformer/LLM models.

We will try using both a self-built/trained decoder, and a fine-tuned Qwen base model; paired with a contrastive visual transformer to create image embedding tokens.

# Set-up

* Install the [git lfs](https://git-lfs.com/) extension **before cloning this repository**
* Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)

Then install dependencies with:

```bash
uv sync --all-packages --dev
```

# Model Training

Run the following, with an optional `--model "model_name"` parameter

```bash
uv run -m model.start_train
```

# Run streamlit app

```bash
uv run streamlit run streamlit/app.py
```
