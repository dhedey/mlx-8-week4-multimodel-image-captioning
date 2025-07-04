from model.modules.bert_custom_multi_modal_model import BertMultiModalModelConfig
from model.modules.qwen_multi_modal_model import QwenMultiModalModelConfig, SpecialTokensStrategy, \
    EmbeddingLearningStrategy
from .models import ImageCaptioningModel, ImageCaptioningModelConfig
from .trainer import ImageCaptioningModelTrainer, ImageCaptioningModelTrainingConfig
import torch

has_cuda = torch.cuda.is_available()

DEFAULT_MODEL_PARAMETERS = {
    "qwen-base-captioner-pirate-v2": {
        "model_class": ImageCaptioningModel,
        "model": ImageCaptioningModelConfig(
            model=QwenMultiModalModelConfig(
                freeze_visual_model=True,
                special_tokens_strategy=SpecialTokensStrategy.CREATE_NEW,
                embedding_learning_strategy=EmbeddingLearningStrategy.LORA,
            )
        ),
        "model_trainer": ImageCaptioningModelTrainer,
        "training": ImageCaptioningModelTrainingConfig(
            # Note: This is a very small batch size for Qwen models, but it is necessary to fit in memory.
            batch_size=4,
            print_after_batches=5,
            validation_max_print_examples=3 if has_cuda else 1,
            epochs=3,
            learning_rate=0.0005,
            optimizer="adamw",
            dataset_kind="pirate",
            save_only_grad_weights=True,
        ),
    },
    "qwen-base-captioner-pirate": {
        "model_class": ImageCaptioningModel,
        "model": ImageCaptioningModelConfig(
            model=QwenMultiModalModelConfig(
                freeze_visual_model=True,
                freeze_new_special_token_embeddings=True,
            )
        ),
        "model_trainer": ImageCaptioningModelTrainer,
        "training": ImageCaptioningModelTrainingConfig(
            # Note: This is a very small batch size for Qwen models, but it is necessary to fit in memory.
            batch_size=8,
            print_after_batches=5,
            validation_max_print_examples=5 if has_cuda else 1,
            epochs=10,
            learning_rate=0.0005,
            optimizer="adamw",
            dataset_kind="pirate",
            save_only_grad_weights=True,
        ),
    },
    "qwen-base-captioner-v1": {
        "model_class": ImageCaptioningModel,
        "model": ImageCaptioningModelConfig(
            model = QwenMultiModalModelConfig(
                freeze_visual_model=True,
                special_tokens_strategy=SpecialTokensStrategy.CREATE_NEW,
                embedding_learning_strategy=EmbeddingLearningStrategy.LORA,
            )
        ),
        "model_trainer": ImageCaptioningModelTrainer,
        "training": ImageCaptioningModelTrainingConfig(
            batch_size=4,
            epochs=1,
            learning_rate=0.0005,
            print_after_batches=5,
            validation_max_print_examples=3 if has_cuda else 1,
            custom_validate_after_batches=200,
            optimizer="adamw",
            dataset_kind="standard",
            save_only_grad_weights=True,
        ),
    },
    "custom-image-captioner-v1": {
        "model_class": ImageCaptioningModel,
        "model": ImageCaptioningModelConfig(
            model = BertMultiModalModelConfig(
                num_layers = 8,
                heads_per_layer = 8,
                attention_kq_dimension = 128,
                attention_v_dimension = 128,
                rope_enabled = True,

                mlp_hidden_dimension = 256 * 4,
                mlp_dropout = 0.2,

                freeze_bert_weights = True,
                freeze_image_weights = True,
            ),
        ),
        "model_trainer": ImageCaptioningModelTrainer,
        "training": ImageCaptioningModelTrainingConfig(
            batch_size=128,
            epochs=10,
            learning_rate=0.001,
            optimizer="adamw",
            dataset_kind="standard",
        ),
    },
}

DEFAULT_MODEL_NAME=list(DEFAULT_MODEL_PARAMETERS.keys())[0]

if __name__ == "__main__":
    import os
    from PIL import Image
    from .common import ModelBase, ModelTrainerBase, TrainingOverrides
    example_image = Image.open(os.path.join(os.path.dirname(__file__), "example.jpg"))

    for model_name, parameters in DEFAULT_MODEL_PARAMETERS.items():
        best_version = f"{model_name}-best"

        if not ModelBase.exists(model_name=best_version):
            print(f"Model {best_version} does not exist locally. Skipping.")
            continue

        print(f"Loading Model: {best_version}")

        model, training_state, training_config = ModelBase.load_advanced(model_name=best_version)
        model = model.eval()

        model.print_detailed_parameter_counts()

        if isinstance(model, ImageCaptioningModel):
            print("Generating caption for example image:")
            with torch.no_grad():
                for token in model.generate_caption_streaming(example_image):
                    print(token, end="", flush=True)
                print()
                print()

        trainer = ModelTrainerBase.load(model, training_config, training_state)

        print(f"Latest validation metrics: {trainer.latest_validation_results}")

        print(f"Running model to check it's working...")
        trainer.custom_validation()
