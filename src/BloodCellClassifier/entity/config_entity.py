## ENTITY
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    EOSINOPHIL_dirs: list
    LYMPHOCYTE_dir: list
    MONOCYTE_dirs: list
    NEUTROPHIL_dirs: list
    img_height: int
    img_width: int
    batch_size: int
    test_size: float
    val_size: float
    seed: int



@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    trained_model_path: Path
    tensorboard_log_dir: Path
    epochs: int
    learning_rate: float
    batch_size: int
    img_height: int
    img_width: int



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    mlflow_uri: str
    batch_size: int
    img_height: int
    img_width: int
    target_metric: str
