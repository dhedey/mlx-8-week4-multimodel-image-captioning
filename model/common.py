from collections import OrderedDict
from dataclasses import dataclass
import torch.optim as optim
import torch.nn as nn
import torch
import os
import time
import wandb
import pathlib
import inspect
import re
from typing import Optional, Self
import pydantic
from pydantic import Field

class PersistableData(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        use_enum_values=True,
        # See note here: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.use_enum_values
        validate_default=True,
    )

    def to_dict(self) -> dict:
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(**d)

class ModuleConfig(PersistableData):
    pass

_selected_device = None

def select_device():
    global _selected_device
    if _selected_device is None:
        DEVICE_IF_MPS_SUPPORT = 'cpu' # or 'mps' - but it doesn't work well with EmbeddingBag
        device = torch.device('cuda' if torch.cuda.is_available() else DEVICE_IF_MPS_SUPPORT if torch.backends.mps.is_available() else 'cpu')
        
        print(f'Selected device: {device}')
        _selected_device = device

    return _selected_device

class TrainingConfig(PersistableData):
    batch_size: int
    epochs: int
    learning_rate: float
    save_only_grad_weights: bool = False
    warmup_epochs: int = 0 # Number of epochs to warm up the learning rate
    print_after_batches: int = 10
    custom_validate_after_batches: Optional[int] = None
    batch_limit: Optional[int] = None
    optimizer: str = "AdamW"
    """
    The optimizer to use for training.
    Currently supports "Adam" and "AdamW".
    In future, it could support any under https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer
    """
    optimizer_params: Optional[dict] = None
    """
    The parameters to use with the optimizer. You don't need to set the learning rate, this is overridden from the learning_rate field.
    """
    schedulers: list[str | tuple[str, dict]] = Field(default_factory=list, description="List of (scheduler_name, scheduler_params) tuples")
    """
    The scheduler/s to use for training along with their parameters.
    The names should be any scheduler name under the `torch.optim.lr_scheduler` namespace, e.g. "LRScheduler"
    If a list of scheduler names is provided, they will be wrapped in a chained scheduler.
    See https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate for more details.
    """
    early_stopping: bool = False
    early_stopping_patience: int = 5

class ValidationResults(PersistableData, extra="allow"):
    epoch: int
    train_comparable_loss: float = 0.0 # Default to 0.0 to support backwards compatibility
    """
    The average training loss is a measure of how well the model performs on the validation set, in comparison to the training set.
    IMPORTANT: This must be comparable to the training loss, to act as a measure of overfitting.
    """
    # This should be some measure of how well the validation went. Low is good.
    # It could simply be the average loss over the validation set, or some other better metric.
    # It should be comparable between epochs, but is not necessarily comparable to the training loss.
    validation_loss: float
    """
    The validation loss is a good measure of how well the model performs on the validation set.
    It could simply be the same as average_training_loss but could be any other better metric, so long as it is comparable between epochs.
    IMPORTANT: It is not necessarily comparable to the training average loss.
    """
    # You can add more fields here simply by adding them to the constructor

@dataclass
class BatchResults:
    total_loss: torch.Tensor # Singleton float tensor
    num_samples: int
    intermediates: dict

class EpochTrainingResults(PersistableData):
    epoch: int
    average_loss: float
    num_samples: int

class FullTrainingResults(PersistableData):
    total_epochs: int
    last_validation: ValidationResults
    best_validation: ValidationResults
    last_training_epoch: EpochTrainingResults
    best_training_epoch: EpochTrainingResults

class TrainingState(PersistableData):
    epoch: int
    optimizer_state: dict
    model_trainer_class_name: str
    total_training_time_seconds: float
    latest_training_results: EpochTrainingResults
    best_training_results: Optional[EpochTrainingResults] = None # None to support backwards compatibility
    all_training_results: list[EpochTrainingResults] = Field(default_factory=list) # Default to support backwards compatibility
    latest_validation_results: ValidationResults
    best_validation_results: Optional[ValidationResults] = None # None to support backwards compatibility
    all_validation_results: list[ValidationResults] = Field(default_factory=list) # Default to support backwards compatibility
    scheduler_state: Optional[dict] = None # None to support backwards compatibility

class TrainingOverrides(PersistableData):
    override_batch_size: Optional[int] = None
    override_batch_limit: Optional[int] = None
    override_to_epoch: Optional[int] = None
    override_learning_rate: Optional[float] = None
    print_after_batches: Optional[int] = None
    validate_after_epochs: int = 1
    seed: int = 42
    use_dataset_cache: bool = True
    print_detailed_parameter_counts: bool = False

class ModelBase(nn.Module):
    registered_types: dict[str, type] = {}
    config_class: type[ModuleConfig] = ModuleConfig

    def __init__(self, model_name: str, config: ModuleConfig):
        super().__init__()
        self.model_name = model_name
        self.config=config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Register the class name in the ModelBase.registered_types dictionary
        if cls.__name__ in ModelBase.registered_types:
            raise ValueError(f"Model {cls.__name__} is a duplicate classname. Use a new class name.")
        ModelBase.registered_types[cls.__name__] = cls

        # Register the config class in cls.config_class
        init_signature = inspect.signature(cls.__init__)
        init_params = init_signature.parameters

        if "config" not in init_params:
            raise ValueError(f"Model {cls.__name__} must have a 'config' parameter in its __init__ method.")

        config_param_class = init_params["config"].annotation

        if not issubclass(config_param_class, ModuleConfig):
            raise ValueError(f"Model {cls.__name__} has a 'config' parameter in its __init__ method called {config_param_class}, but this class does not derive from ModuleConfig.")

        cls.config_class = config_param_class

    def get_device(self):
        return next(self.parameters()).device
    
    @staticmethod
    def model_path(model_name: str) -> str:
        model_folder = os.path.join(os.path.dirname(__file__), "saved")
        return os.path.join(model_folder, f"{model_name}.pt")

    def save_model_data(
            self,
            training_config: TrainingConfig,
            training_state: TrainingState,
            file_name: Optional[str] = None,
        ):
        if file_name is None:
            file_name = self.model_name

        model_path = ModelBase.model_path(file_name)
        pathlib.Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)

        if training_config.save_only_grad_weights:
            model_weights = OrderedDict()
            for name, param in self.named_parameters():
                if param.requires_grad:
                    model_weights[name] = param
        else:
            model_weights = self.state_dict()

        torch.save({
            "model": {
                "class_name": type(self).__name__,
                "model_name": self.model_name,
                "weights": model_weights,
                "config": self.config.to_dict(),
            },
            "training": {
                "config": training_config.to_dict(),
                "state": training_state.to_dict(),
            },
        }, model_path)

        print(f"Model saved to {model_path} (save_only_grad_weights={training_config.save_only_grad_weights})")

    @classmethod
    def exists(
        cls,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> bool:
        return os.path.exists(cls.resolve_path(model_name=model_name, model_path=model_path))
    
    @classmethod
    def resolve_path(
        cls,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> str:
        if model_path is not None:
            return model_path
        if model_name is not None:
            return ModelBase.model_path(model_name)
        raise ValueError("Either model_name or model_path must be provided to load a model.")

    @classmethod
    def load_only_training_state(
        cls,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> TrainingState:
        model_path = cls.resolve_path(model_name=model_name, model_path=model_path)
        loaded_model_data = torch.load(model_path)
        return TrainingState.from_dict(loaded_model_data["training"]["state"])

    @classmethod
    def load_advanced(
        cls,
        model_name: Optional[str] = None,
        override_class_name = None,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> tuple[Self, TrainingState, dict]:
        if device is None:
            device = select_device()

        model_path = cls.resolve_path(model_name=model_name, model_path=model_path)

        loaded_model_data = torch.load(model_path, map_location=device)
        print(f"Model data read from {model_path}")

        if model_name is None:
            model_name = loaded_model_data["model"]["model_name"]
    
        loaded_class_name = loaded_model_data["model"]["class_name"]
        actual_class_name = override_class_name if override_class_name is not None else loaded_class_name

        registered_types = ModelBase.registered_types
        if actual_class_name not in registered_types:
            raise ValueError(f"Model class {actual_class_name} is not a known Model. Available classes: {list(registered_types.keys())}")
        model_class: type[Self] = registered_types[actual_class_name]

        if not issubclass(model_class, cls):
            raise ValueError(f"The model {model_name} was attempted to be loaded with {cls.__name__}.load(\"{model_name}\") (loaded class name = {loaded_class_name}, override class name = {override_class_name}), but {model_class} is not a subclass of {cls}.")

        model_weights = loaded_model_data["model"]["weights"]
        model_config = model_class.config_class.from_dict(loaded_model_data["model"]["config"])
        training_state = TrainingState.from_dict(loaded_model_data["training"]["state"])
        training_config = loaded_model_data["training"]["config"]

        model: ModelBase = model_class(
            model_name=model_name,
            config=model_config,
        )
        if training_config.get("save_only_grad_weights", False):
            model.load_state_dict(model_weights, strict=False)
        else:
            model.load_state_dict(model_weights)

        return model.to(device), training_state, training_config

    @classmethod
    def load_for_evaluation(cls, model_name: Optional[str] = None, model_path: Optional[str] = None, device: Optional[str] = None) -> Self:
        model, _, _ = cls.load_advanced(model_name=model_name, model_path=model_path, device=device)
        model.eval()
        return model

    def print_detailed_parameter_counts(self) -> None:
        normalized_parameter_counts = {}
        learnable_weights_count = 0
        total_weights_count = 0
        for name, parameter in self.named_parameters():
            total_weights_count += parameter.numel()
            if not parameter.requires_grad:
                continue
            learnable_weights_count += parameter.numel()
            normalized_name = re.sub(r'\d+', '*', name)
            if normalized_name in normalized_parameter_counts:
                normalized_parameter_counts[normalized_name]["instances"] += 1
                normalized_parameter_counts[normalized_name]["weights_count"] += parameter.numel()
            else:
                normalized_parameter_counts[normalized_name] = {
                    "instances": 1,
                    "weights_count": parameter.numel(),
                }
        sorted_params = sorted(
            normalized_parameter_counts.items(),
            key=lambda item: item[1]["weights_count"],
            reverse=True
        )
        
        print(f"This {self.__class__.__name__} named \"{self.model_name}\" has {learnable_weights_count:,} learnable weights of {total_weights_count:,} total weights, spread as follows:")
        for name, params in sorted_params:
            print(f"- {name}: {params["weights_count"]:,} across {params["instances"]} instances")
        print()

def create_composite_scheduler(
    optimizer: optim.Optimizer,
    schedulers: list[str | tuple[str, dict]],
) -> Optional[optim.lr_scheduler.LRScheduler]:
    def map_scheduler(scheduler: str | tuple[str, dict]) -> optim.lr_scheduler.LRScheduler:
        if isinstance(scheduler, str):
            return create_scheduler(optimizer, scheduler, {})
        else:
            return create_scheduler(optimizer, scheduler[0], scheduler[1])

    match len(schedulers):
        case 0:
            return None
        case 1:
            return map_scheduler(schedulers[0])
        case _:
            # This doesn't work with `ReduceLROnPlateau`...
            # Maybe we could create our own ChainedScheduler where it does work, by intelligently passing through the metrics field
            return optim.lr_scheduler.ChainedScheduler([
                map_scheduler(scheduler) for scheduler in schedulers
            ])

def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    scheduler_params: dict,
) -> optim.lr_scheduler.LRScheduler:
    match scheduler_type:
        case "LRScheduler":
            return optim.lr_scheduler.LRScheduler(optimizer, **scheduler_params)
        case "LambdaLR":
            return optim.lr_scheduler.LambdaLR(optimizer, **scheduler_params)
        case "MultiplicativeLR":
            return optim.lr_scheduler.MultiplicativeLR(optimizer, **scheduler_params)
        case "StepLR":
            return optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        case "MultiStepLR":
            return optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)
        case "LinearLR":
            return optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
        case "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
        case "PolynomialLR":
            return optim.lr_scheduler.PolynomialLR(optimizer, **scheduler_params)
        case "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        case "SequentialLR":
            return optim.lr_scheduler.SequentialLR(optimizer, **scheduler_params)
        case "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
        case "CyclicLR":
            return optim.lr_scheduler.CyclicLR(optimizer, **scheduler_params)
        case "OneCycleLR":
            return optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
        case "CosineAnnealingWarmRestarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
        case _:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def create_optimizer(
    model: ModelBase,
    optimizer_name: str,
    optimizer_params: Optional[dict],
    learning_rate: float,
) -> optim.Optimizer:
    if optimizer_params is None:
        optimizer_params = {}

    optimizer_params["lr"] = learning_rate

    match optimizer_name.lower():
        case "adam":
            return optim.Adam(model.parameters(), **optimizer_params)
        case "adamw":
            return optim.AdamW(model.parameters(), **optimizer_params)
        case _:
            raise ValueError(f"Unsupported optimizer type: {optimizer_name}")

class ModelTrainerBase:
    registered_types: dict[str, type[Self]] = {}
    config_class: type[TrainingConfig] = TrainingConfig

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ in ModelTrainerBase.registered_types:
            raise ValueError(f"ModelTrainer {cls.__name__} is a duplicate classname. Use a new class name.")
        ModelTrainerBase.registered_types[cls.__name__] = cls

        # Register the config class in cls.config_class
        init_signature = inspect.signature(cls.__init__)
        init_params = init_signature.parameters

        if "config" not in init_params:
            raise ValueError(f"ModelTrainer {cls.__name__} must have a 'config' parameter in its __init__ method.")

        config_param_class = init_params["config"].annotation

        if not issubclass(config_param_class, TrainingConfig):
            raise ValueError(f"ModelTrainer {cls.__name__} has a 'config' parameter in its __init__ method called {config_param_class}, but this class does not derive from TrainingConfig.")

        cls.config_class = config_param_class

    def __init__(
            self,
            model: ModelBase,
            config: TrainingConfig,
            continuation: Optional[TrainingState] = None,
            overrides: Optional[TrainingOverrides] = None,
        ):

        self.model = model
        self.config = config

        learnable_weights_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_weights_count = sum(p.numel() for p in model.parameters())
        print(f"Initializing {self.__class__.__name__} for {model.__class__.__name__} named \"{model.model_name}\" (learnable weights = {learnable_weights_count:,} total weights = {total_weights_count:,})")
        print()

        if overrides is None:
            overrides = TrainingOverrides()

        torch.manual_seed(overrides.seed)

        if overrides.print_detailed_parameter_counts:
            self.model.print_detailed_parameter_counts()

        self.validate_after_epochs = overrides.validate_after_epochs

        if overrides.override_to_epoch is not None:
            self.config.epochs = overrides.override_to_epoch
            print(f"Overriding training end epoch to {self.config.epochs}")
        
        if overrides.override_batch_size is not None:
            self.config.batch_size = overrides.override_batch_size
            print(f"Overriding batch size to {self.config.batch_size}")

        if overrides.override_batch_limit is not None:
            self.config.batch_limit = overrides.override_batch_limit
            print(f"Overriding batch limit to {self.config.batch_limit}")

        if overrides.override_learning_rate is not None:
            self.config.learning_rate = overrides.override_learning_rate
            print(f"Overriding learning rate to {self.config.learning_rate}")

        if overrides.print_after_batches is not None:
            self.config.print_after_batches = overrides.print_after_batches
            print(f"Overriding print after batches to {self.config.print_after_batches}")

        self.optimizer = create_optimizer(self.model, self.config.optimizer, self.config.optimizer_params, self.config.learning_rate)

        schedulers = self.config.schedulers.copy()
        if config.warmup_epochs > 0:
            print(f"{config.warmup_epochs} warm-up epochs were defined, so configuring an initial linear scheduler")
            if len(schedulers) > 0 and isinstance(schedulers[0], tuple) and schedulers[0][0] == "LinearLR":
                # We already added a scheduler and persisted it (before there was a copy on the line above)
                pass
            else:
                schedulers.insert(0, ("LinearLR", { "total_iters": config.warmup_epochs }))

        self.scheduler = create_composite_scheduler(self.optimizer, schedulers)

        if continuation is not None:
            self.epoch = continuation.epoch
            self.optimizer.load_state_dict(continuation.optimizer_state)
            if overrides.override_learning_rate is not None:
                # After loading the state dict, we need to set the override again
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = overrides.override_learning_rate
            if self.scheduler is not None and continuation.scheduler_state is not None:
                self.scheduler.load_state_dict(continuation.scheduler_state)
            self.total_training_time_seconds = continuation.total_training_time_seconds
            self.latest_training_results = continuation.latest_training_results
            self.best_training_results = continuation.best_training_results if continuation.best_training_results is not None else continuation.latest_training_results
            self.all_training_results = continuation.all_training_results
            self.latest_validation_results = continuation.latest_validation_results
            self.best_validation_results = continuation.best_validation_results if continuation.best_validation_results is not None else continuation.latest_validation_results
            self.all_validation_results = continuation.all_validation_results
            print(f"Resuming training from saved state after epoch {self.epoch}")
            print()
        else:
            self.epoch = 0
            self.total_training_time_seconds = 0.0
            self.latest_training_results = None
            self.best_training_results = None
            self.all_training_results = []
            self.latest_validation_results = None
            self.best_validation_results = None
            self.all_validation_results = []

    @property
    def train_data_loader(self):
        raise NotImplementedError("This property should be implemented by subclasses.")

    @property
    def validation_data_loader(self):
        raise NotImplementedError("This property should be implemented by subclasses.")

    def process_batch(self, raw_batch) -> BatchResults:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def custom_validation(self) -> Optional[dict]:
        """
        This allows trainers to add specific validations.
        If they have a better measure of validation success, they should return "validation_loss" in their dictionary,
        which will override the train_comparable_loss
        """
        print("No custom validations have been defined in the trainer, skipping")
        print()
        return None

    @classmethod
    def load_with_model(cls, model_name: Optional[str] = None, overrides: Optional[TrainingOverrides] = None, device: Optional[str] = None, model_path: Optional[str] = None) -> Self:
        model, state, config = ModelBase.load_advanced(model_name=model_name, device=device, model_path=model_path)
        return cls.load(
            model=model,
            config=config,
            state=state,
            overrides=overrides,
        )

    @classmethod
    def load(cls, model: ModelBase, config: dict, state: TrainingState, overrides: Optional[TrainingOverrides] = None) -> Self:
        trainer_class_name = state.model_trainer_class_name

        registered_types = ModelTrainerBase.registered_types
        if trainer_class_name not in registered_types:
            raise ValueError(f"Trainer class {trainer_class_name} is not a known ModelTrainer. Available classes: {list(registered_types.keys())}")
        trainer_class: type[Self] = registered_types[trainer_class_name]

        if not issubclass(trainer_class, cls):
            raise ValueError(f"The trainer was attempted to be loaded with {cls.__name__}.load(..) with a trainer class of \"{trainer_class_name}\"), but {trainer_class_name} is not a subclass of {cls}.")
        
        config = trainer_class.config_class.from_dict(config)

        trainer = trainer_class(
            model=model,
            config=config,
            continuation=state,
            overrides=overrides,
        )

        return trainer

    def train(self) -> FullTrainingResults:
        while self.epoch < self.config.epochs:
            self.epoch += 1
            print(f"======================== EPOCH {self.epoch}/{self.config.epochs} ========================")
            print()
            self.train_epoch()
            if self.epoch % self.validate_after_epochs == 0 or self.epoch == self.config.epochs or self.latest_validation_results is None:
                self.validate()

                if wandb.run is not None:
                    log_data = {
                        "epoch": self.epoch,
                    }
                    def add_prefixed(data: dict, new_data: dict, prefix: str):
                        for key, value in new_data.items():
                            if key == "epoch":
                                continue
                            if key.startswith(prefix):
                                data[key] = value
                            else:
                                data[f"{prefix}{key}"] = value
                    add_prefixed(log_data, self.latest_validation_results.to_dict(), "validation_")
                    add_prefixed(log_data, self.latest_training_results.to_dict(), "train_")
    
                    wandb.log(log_data)
            else:
                print(f"Skipping validation after epoch {self.epoch} because validate_after_epochs is set to {self.validate_after_epochs}, and it's not the first or last epoch")

            if self.scheduler is not None:
                print(f"Stepping scheduler at end of epoch {self.epoch}")

                # Handle the parameters for the ReduceLROnPlateau scheduler
                scheduler_step_parameters = inspect.signature(self.scheduler.step).parameters
                if "metrics" in scheduler_step_parameters.keys():
                    self.scheduler.step(self.latest_validation_results.validation_loss)
                else:
                    self.scheduler.step()

            self.save_model()
            print()

            if self.config.early_stopping:
                best_results_epochs_ago = self.best_validation_results.epoch - self.latest_validation_results.epoch

                if best_results_epochs_ago > self.config.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {self.latest_validation_results.epoch}, because the best validation results occurred at epoch {self.best_validation_results.epoch}, over {self.config.early_stopping_patience} epochs ago")
                    break

        print("Training complete.")

        return FullTrainingResults(
            total_epochs=self.config.epochs,
            last_validation=self.latest_validation_results,
            best_validation=self.best_validation_results,
            last_training_epoch=self.latest_training_results,
            best_training_epoch=self.best_training_results,
        )

    def train_epoch(self):
        self.model.train()

        print_every = self.config.print_after_batches
        custom_validate_every = self.config.custom_validate_after_batches

        running_loss = 0.0
        running_samples = 0

        total_batches = len(self.train_data_loader)
        
        start_epoch_time_at = time.time()

        if self.config.batch_limit is not None and total_batches >= self.config.batch_limit:
            total_batches_text = f"{self.config.batch_limit:,} (limited from {total_batches:,} by config.batch_limit)"
            total_batches = self.config.batch_limit
        else:
            total_batches_text = f"{total_batches:,}"

        print(f"> Starting training... (batch_size={self.config.batch_size}, total_batches={total_batches_text})")
        print()

        epoch_loss = 0.0
        epoch_samples = 0
        for batch_idx, raw_batch in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            batch_results = self.process_batch(raw_batch)

            loss = batch_results.total_loss
            running_samples += batch_results.num_samples
            running_loss += loss.item()
            epoch_samples += batch_results.num_samples
            epoch_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            batch_num = batch_idx + 1
            if batch_num % print_every == 0 or batch_num == total_batches:
                print(f"Epoch {self.epoch}/{self.config.epochs}, Batch {batch_num}/{total_batches}, Recent Avg Loss: {(running_loss / running_samples):.3g}")
                running_loss = 0.0
                running_samples = 0

            if custom_validate_every is not None and batch_num % custom_validate_every == 0:
                print()
                print(f"Starting mid-epoch custom validation at batch {batch_num}:")
                print()
                self.custom_validation()

            if batch_num == total_batches: # Handle self.config.batch_limit
                break

        average_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        training_time = time.time() - start_epoch_time_at
        print()
        print(f"Epoch training complete (Epoch Avg Loss: {average_loss:.3g}, Time: {training_time:.1f}s)")
        print()

        if self.total_training_time_seconds is not None:
            self.total_training_time_seconds += training_time

        training_results = EpochTrainingResults(
            epoch=self.epoch,
            average_loss=average_loss,
            num_samples=epoch_samples,
        )
        self.latest_training_results = training_results

        if len(self.all_training_results) == 0 or self.all_training_results[-1].epoch < training_results.epoch:
            self.all_training_results.append(training_results)

        if self.best_training_results is None or training_results.average_loss < self.best_training_results.average_loss:
            self.best_training_results = training_results

    def train_comparable_validation(self) -> ValidationResults:
        print_every = self.config.print_after_batches
        running_loss = 0.0
        running_samples = 0

        total_batches = len(self.train_data_loader)
        start_time = time.time()

        if self.config.batch_limit is not None and total_batches >= self.config.batch_limit:
            total_batches_text = f"{self.config.batch_limit:,} (limited from {total_batches:,} by config.batch_limit)"
            total_batches = self.config.batch_limit
        else:
            total_batches_text = f"{total_batches:,}"

        print(f"> Starting train-comparable validation... (batch_size={self.config.batch_size}, total_batches={total_batches_text})")
        print()

        total_loss = 0.0
        total_samples = 0
        for batch_idx, raw_batch in enumerate(self.validation_data_loader):
            batch_results = self.process_batch(raw_batch)

            loss = batch_results.total_loss
            running_samples += batch_results.num_samples
            running_loss += loss.item()
            total_samples += batch_results.num_samples
            total_loss += loss.item()

            batch_num = batch_idx + 1
            if batch_num % print_every == 0 or batch_num == total_batches:
                print(f"Validation batch {batch_num}/{total_batches}, Recent Avg Loss: {(running_loss / running_samples):.3g}")
                running_loss = 0.0
                running_samples = 0

            if batch_num == total_batches:  # Handle self.config.batch_limit
                break

        average_loss = total_loss / total_samples if total_samples > 0 else 0.0

        validation_results = ValidationResults(
            epoch=self.epoch,
            train_comparable_loss=average_loss,
            validation_loss=average_loss,
        )

        self.latest_validation_results = validation_results
        if len(self.all_validation_results) == 0 or self.all_validation_results[-1].epoch < validation_results.epoch:
            self.all_validation_results.append(validation_results)
        if self.best_validation_results is None or validation_results.validation_loss < self.best_validation_results.validation_loss:
            self.best_validation_results = validation_results

        time_elapsed = time.time() - start_time
        validation_average_train_loss = self.latest_validation_results.train_comparable_loss
        if self.latest_training_results is not None:
            train_average_loss = self.latest_training_results.average_loss
            overfitting_measure = (validation_average_train_loss - train_average_loss) / validation_average_train_loss if validation_average_train_loss > 0 else float('inf')
        else:
            train_average_loss = "UNK"
            overfitting_measure = "UNK"

        print()
        print(f"Train-comparable validation complete (Val/Train Loss: {validation_average_train_loss:.3g}/{train_average_loss:.3g}, Overfitting: {overfitting_measure:.2%}, Time: {time_elapsed:.1f}s)")
        print()

        return validation_results

    def validate(self) -> ValidationResults:
        self.model.eval()

        with torch.no_grad():

            print("> Starting custom validation...")
            print()
            custom_results = self.custom_validation()
            print()

            validation_results = self.train_comparable_validation()
            if custom_results is None:
                custom_results = {}

            for key, value in custom_results.items():
                validation_results.key = value

        return validation_results

    def save_model(self):
        scheduler_state = self.scheduler.state_dict() if self.scheduler is not None else None
        training_state = TrainingState(
            epoch=self.epoch,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=scheduler_state,
            model_trainer_class_name=self.__class__.__name__,
            total_training_time_seconds=self.total_training_time_seconds,
            latest_training_results=self.latest_training_results,
            all_training_results=self.all_training_results,
            best_training_results=self.best_training_results,
            latest_validation_results=self.latest_validation_results,
            best_validation_results=self.best_validation_results,
            all_validation_results=self.all_validation_results,
        )
        self.model.save_model_data(
            file_name=self.model.model_name,
            training_config=self.config,
            training_state=training_state
        )

        if self.latest_validation_results is None:
            print("No new validation loss available, skipping comparison with the best model.")
            return

        best_model_name = self.model.model_name + '-best'
        best_validation_loss = None
        best_validation_epoch = None
        if ModelBase.exists(model_name=best_model_name):
            try:
                best_training_state = ModelBase.load_only_training_state(model_name=best_model_name)
                best_validation_loss = best_training_state.latest_validation_results.validation_loss
                best_validation_epoch = best_training_state.latest_validation_results.epoch
            except Exception as e:
                print(f"⚠️ Failed to load the best model training state: {e}")

        def format_optional_float(value):
            return f"{value:.3g}" if value is not None else "N/A"

        latest_validation_loss = self.latest_validation_results.validation_loss

        is_improvement = best_validation_loss is None or latest_validation_loss < best_validation_loss
        if is_improvement:
            print(f"The current validation loss {format_optional_float(latest_validation_loss)} is better than the previous best validation loss {format_optional_float(best_validation_loss)} from epoch {best_validation_epoch}, saving as {best_model_name}...")
            self.model.save_model_data(
                file_name=best_model_name,
                training_config=self.config,
                training_state=training_state,
            )
        else:
            print(f"The current validation loss {format_optional_float(latest_validation_loss)} is not better than the previous best validation loss {format_optional_float(best_validation_loss)} from epoch {best_validation_epoch}, so not saving as best.")

def upload_model_artifact(
    model_name: str,
    file_path: str,
    artifact_name: str,
    metadata: dict = None,
    description: str = None
):
    """
    Upload a model as a wandb artifact.
    
    Args:
        model_name: Name of the model. Used for the file name inside the artifact.
        model_path: Path to the saved model file
        artifact_name: Optional custom artifact name (defaults to model_name)
        metadata: Optional metadata dictionary to include with the artifact
        description: Optional description for the artifact
    """
    if not os.path.exists(file_path):
        print(f"⚠️ Model file not found: {file_path}")
        return None
    
    # Create artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=description or f"Trained model: {model_name}",
        metadata=metadata or {}
    )
    
    # Add the model file
    artifact.add_file(file_path, name=f"{model_name}.pt")
    
    # Log the artifact
    wandb.log_artifact(artifact)
    print(f"📦 Uploaded model artifact: {artifact_name}")
    
    return artifact
