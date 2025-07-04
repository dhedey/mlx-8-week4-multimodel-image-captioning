# Run as uv run -m model.continue_train
import argparse
from .default_models import DEFAULT_MODEL_NAME
from .wandb_config import WANDB_PROJECT_NAME, WANDB_ENTITY
from .trainer import ModelTrainerBase, TrainingOverrides
from .common import upload_model_artifact, ModelBase
import wandb
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continue training a model')
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
        '--model-source',
        choices=["default", "local", "wandb"],
        default="default",
        help='Enable wandb logging'
    )
    parser.add_argument(
        '--wandb-continue-run',
        type=str,
        default=None,
        help='Continue from a wandb run ID, enabling W&B logging.'
    )
    parser.add_argument(
        '--wandb-artifact-name',
        type=str,
        default=None,
        help='Continue from a specific artifact name. This is read from the run config.'
    )
    parser.add_argument(
        '--wandb-artifact-type',
        type=str,
        choices=["best", "final"],
        default="best",
        help='Which model to load. Not used if the artifact name is specified.'
    )
    parser.add_argument(
        '--wandb-artifact-version',
        type=str,
        default="latest",
        help='The version of the artifact to load. Not used if the artifact name is specified.'
    )
    parser.add_argument(
        '--wandb-project', 
        default=WANDB_PROJECT_NAME, 
        help='W&B project name (used if --from-wandb is set)'
    )
    parser.add_argument(
        '--wandb-entity',
        default=WANDB_ENTITY,
        help='W&B entity name (used if --from-wandb is set)'
    )
    parser.add_argument(
        '--no-upload-artifacts', 
        action='store_true', 
        help='Disable artifact uploading to W&B (if --from-wandb is set)'
    )
    args = parser.parse_args()

    overrides = TrainingOverrides(
        override_to_epoch=args.end_epoch,
        override_batch_size=args.batch_size,
        override_batch_limit=args.batch_limit,
        override_learning_rate=args.learning_rate,
        validate_after_epochs=args.validate_after_epochs,
        print_after_batches=args.print_after_batches,
        seed=args.seed,
        use_dataset_cache=not args.ignore_dataset_cache,
    )

    try:
        use_wandb = args.wandb or args.wandb_continue_run is not None or args.wandb_artifact_name is not None

        match args.model_source:
            case "default":
                if use_wandb:
                    model_source = "wandb"
                else:
                    model_source = "local"
            case "wandb":
                model_source = "wandb"
            case "local":
                model_source = "local"
            case _:
                raise ValueError(f"Unknown model_source {args.model_source}")

        if use_wandb and args.wandb_continue_run is not None:
            run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                id=args.wandb_continue_run,
                resume="must"
            )
            print(f"🏃 Resumed wandb run {args.wandb_continue_run}")

        match model_source:
            case "local":
                model_source_run = None
                model_file_path = ModelBase.model_path(model_name=args.model)
            case "wandb":
                if args.wandb_artifact_name:
                    artifact_name = args.wandb_artifact_name
                    print(f"Loading provided artifact name: {artifact_name}")
                elif wandb.run is not None:
                    model_name = wandb.run.config["model_name"]
                    artifact_name = f"{model_name}-{args.wandb_artifact_type}"
                    print(f"Loading artifact name from the continued run model name: {artifact_name}")
                else:
                    model_name = args.model
                    artifact_name = f"{model_name}-{args.wandb_artifact_type}"
                    print(f"Assuming artifact name from the provided model name: {artifact_name}")

                full_artifact_identifier = f"{args.wandb_entity}/{args.wandb_project}/{artifact_name}:{args.wandb_artifact_version}"
                try:
                    if wandb.run is not None:
                        artifact = wandb.run.use_artifact(full_artifact_identifier)
                    else:
                        api = wandb.Api()
                        artifact = api.artifact(full_artifact_identifier)
                except wandb.errors.CommError as e:
                    print(f"Error: Could not find wandb artifact '{full_artifact_identifier}''. {e}")
                    exit(1)

                model_source_run = artifact.logged_by()
                download_path = artifact.download()
                pt_files = [f for f in os.listdir(download_path) if f.endswith('.pt')]
                if not pt_files:
                    print(f"Error: No .pt file found in downloaded artifact at {download_path}")
                    exit(1)
                model_file_path = os.path.join(download_path, pt_files[0])
                print(f"Found downloaded model file: {model_file_path}")
            case _:
                raise ValueError(f"Unknown model_source {model_source}")

        trainer = ModelTrainerBase.load_with_model(
            model_path=model_file_path,
            overrides=overrides,
        )

        if args.immediate_validation:
            print("Immediate validation enabled, running validation before training:")
            trainer.validate()

        if trainer.epoch >= trainer.config.epochs:
            raise ValueError(f"Model {trainer.model.model_name} has already finished training its {trainer.config.epochs} epochs. Use --end-epoch <new_total_epochs> to continue training.")

        model_name = trainer.model.model_name

        if use_wandb and wandb.run is None:
            # Start a new W&B run for the continued training
            new_run_id = wandb.util.generate_id()

            config = {
                "model_name": trainer.model.model_name,
                "model_config": trainer.model.config.to_dict(),
                "training_config": trainer.config.to_dict(),
                "overrides": overrides.to_dict(),
                "from_epoch": trainer.epoch,
            }

            if model_source_run is not None:
                new_run_name = f"{model_source_run.id}-cont-{new_run_id}"
                config["source_run_id"] = model_source_run.id
            else:
                new_run_name = f"train-{model_name}-{new_run_id}"

            wandb.init(
                id=new_run_id,
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=new_run_name,
                config=config,
            )
            print(f"🏃 Started wandb run {wandb.run.id}")

        results = trainer.train()

        if wandb.run is not None and not args.no_upload_artifacts:
            print("Uploading artifacts...")
            model_name = trainer.model.model_name
            artifact_metadata = {
                "model_config": trainer.model.config.to_dict(),
                "training_config": trainer.config.to_dict(),
                "final_validation_loss": results.last_validation.validation_loss,
                "final_train_loss": results.last_training_epoch.average_loss,
                "total_epochs": results.total_epochs,
            }

            if model_source_run is not None:
                artifact_metadata["source_run_id"] = model_source_run.id

            model_path = ModelBase.model_path(model_name)
            best_model_path = ModelBase.model_path(f"{model_name}-best")

            if os.path.exists(model_path):
                upload_model_artifact(
                    model_name=model_name,
                    file_path=model_path,
                    artifact_name=f"{model_name}-final",
                    metadata=artifact_metadata,
                    description=f"Final model from continued run {wandb.run.id}"
                )

            if os.path.exists(best_model_path):
                best_metadata = artifact_metadata.copy()
                best_metadata["model_type"] = "best_validation"
                upload_model_artifact(
                    model_name=model_name,
                    file_path=best_model_path,
                    artifact_name=f"{model_name}-best",
                    metadata=best_metadata,
                    description=f"Best validation model from continued run {wandb.run.id}"
                )
    finally:
        if wandb.run is not None:
            wandb.finish()



        
