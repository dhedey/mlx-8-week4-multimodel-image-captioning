# Run as uv run -m model.start_train

import argparse
from .common import select_device, ModelBase, upload_model_artifact, TrainingOverrides
from .default_models import DEFAULT_MODEL_PARAMETERS, DEFAULT_MODEL_NAME
from .wandb_config import WANDB_PROJECT_NAME, WANDB_ENTITY
import wandb
import os

if __name__ == "__main__":
    device = select_device()

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        '--end-epoch',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size for training (default: None, uses model default)'
    )
    parser.add_argument(
        '--batch-limit',
        type=int,
        default=None,
        help='Override the batch count per epoch/validation (default: None, uses full batch)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Override the default learning rate for training (default: None, uses model default)'
    )
    parser.add_argument(
        '--ignore-dataset-cache',
        action='store_true',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Set a random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        '--print-after-batches',
        type=int,
        default=None,
        help="Print a log message after N batches (default: None, uses model default)",
    )
    parser.add_argument(
        '--validate-after-epochs',
        type=int,
        default=1,
        help="Run validation after every N epochs (default: 1)",
    )
    parser.add_argument(
        '--immediate-validation',
        action='store_true',
        help='Run validation immediately after loading the model'
    )
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping during training'
    )
    parser.add_argument(
        '--wandb', 
        action='store_true', 
        help='Enable wandb logging'
    )
    parser.add_argument(
        '--wandb-project', 
        default=WANDB_PROJECT_NAME, 
        help=f'W&B project name (default: {WANDB_PROJECT_NAME})'
    )
    parser.add_argument(
        '--wandb-entity',
        default=WANDB_ENTITY,
        help='W&B entity name (used if --from-wandb is set)'
    )
    parser.add_argument(
        '--no-upload-artifacts',
        action='store_true',
        help='Disable artifact uploading to W&B (if wandb is enabled)'
    )
    
    args = parser.parse_args()

    model_name = args.model

    if model_name not in DEFAULT_MODEL_PARAMETERS:
        raise ValueError(f"Model '{model_name}' is not defined. The choices are: {list(DEFAULT_MODEL_PARAMETERS.keys())}")

    parameters = DEFAULT_MODEL_PARAMETERS[model_name]

    if args.wandb:
        run_id = wandb.util.generate_id()
        wandb.init(
            id=run_id,
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"train-{model_name}-{run_id}",
            config={
                "model_name": model_name,
                "model_config": parameters["model"].to_dict(),
                "training_config": parameters["training"].to_dict(),
                "from_epoch": 0,
            },
        )

    model = parameters["model_class"](
        model_name=model_name,
        config=parameters["model"],
    ).to(device)

    overrides = TrainingOverrides(
        print_detailed_parameter_counts=True,
        override_to_epoch=args.end_epoch,
        override_batch_size=args.batch_size,
        override_batch_limit=args.batch_limit,
        override_learning_rate=args.learning_rate,
        validate_after_epochs=args.validate_after_epochs,
        print_after_batches=args.print_after_batches,
        seed=args.seed,
        use_dataset_cache=not args.ignore_dataset_cache,
    )
    trainer = parameters["model_trainer"](
        model=model,
        config=parameters["training"],
        overrides=overrides,
    )

    if args.immediate_validation:
        print("Immediate validation enabled, running validation before training:")
        trainer.validate()

    results = trainer.train()

    if wandb.run is not None and not args.no_upload_artifacts:
        print("Uploading artifacts...")

        artifact_metadata = {
            "model_name": model_name,
            "model_config": parameters["model"].to_dict(),
            "training_config": parameters["training"].to_dict(),
            "final_validation_loss": results.last_validation.validation_loss,
            "final_train_loss": results.last_training_epoch.average_loss,
            "total_epochs": results.total_epochs,
        }
        
        model_path = ModelBase.model_path(model_name)
        best_model_path = ModelBase.model_path(f"{model_name}-best")

        if os.path.exists(model_path):
            upload_model_artifact(
                model_name=model_name,
                file_path=model_path,
                artifact_name=f"{model_name}-final",
                metadata=artifact_metadata,
                description=f"Final model: {model_name}"
            )

        if os.path.exists(best_model_path):
            best_metadata = artifact_metadata.copy()
            best_metadata["model_type"] = "best_validation"
            upload_model_artifact(
                model_name=model_name,
                file_path=best_model_path,
                artifact_name=f"{model_name}-best",
                metadata=best_metadata,
                description=f"Best validation model: {model_name}"
            )

    if args.wandb:
        wandb.finish()



        
