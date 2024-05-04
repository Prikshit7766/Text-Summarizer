from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: Path
    ALL_REQUIRED_FILLS: list


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str
    
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    output_dir: Path
    evaluation_strategy: str
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    warmup_steps: int
    weight_decay: float
    save_total_limit: int
    gradient_accumulation_steps: int
    logging_dir: Path
    save_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    fp16: bool    