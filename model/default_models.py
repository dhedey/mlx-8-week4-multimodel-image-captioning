from model.modules.bert_custom_multi_modal_model import BertMultiModalModelConfig
from model.modules.qwen_multi_modal_model import QwenMultiModalModelConfig
from .models import ImageCaptioningModel, ImageCaptioningModelConfig
from .trainer import ModelTrainerBase, ImageCaptioningModelTrainer, ImageCaptioningModelTrainingConfig

DEFAULT_MODEL_PARAMETERS = {
    "qwen-base-captioner-pirate": {
        "model_class": ImageCaptioningModel,
        "model": ImageCaptioningModelConfig(
            model=QwenMultiModalModelConfig(
                freeze_visual_model=True,
            )
        ),
        "model_trainer": ImageCaptioningModelTrainer,
        "training": ImageCaptioningModelTrainingConfig(
            batch_size=128,
            epochs=10,
            learning_rate=0.001,
            optimizer="adamw",
            dataset_kind="pirate",
        ),
    },
    "qwen-base-captioner-v1": {
        "model_class": ImageCaptioningModel,
        "model": ImageCaptioningModelConfig(
            model = QwenMultiModalModelConfig(
                freeze_visual_model=True,
            )
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
   for model_name, parameters in DEFAULT_MODEL_PARAMETERS.items():
        best_version = f"{model_name}-best"
        print(f"Loading Model: {best_version}")

        trainer = ModelTrainerBase.load_with_model(best_version)
        print(f"Latest validation metrics: {trainer.latest_validation_results}")
   
        print(f"Running model to check it's working...")
        trainer.run_validation()