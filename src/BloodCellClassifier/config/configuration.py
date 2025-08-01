from src.BloodCellClassifier.constant import *
from src.BloodCellClassifier.utils.common import read_yaml,create_directories 
from src.BloodCellClassifier.entity.config_entity import DataIngestionConfig,DataTransformationConfig,ModelTrainingConfig,ModelEvaluationConfig
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    


    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            EOSINOPHIL_dirs=[Path(x) for x in config.EOSINOPHIL_dirs],
            LYMPHOCYTE_dir=[Path(x) for x in config.LYMPHOCYTE_dir],
            MONOCYTE_dirs=[Path(x) for x in config.MONOCYTE_dirs],
            NEUTROPHIL_dirs=[Path(x) for x in config.NEUTROPHIL_dirs],
            img_height=self.params.IMG_HEIGHT,
            img_width=self.params.IMG_WIDTH,
            batch_size=self.params.BATCH_SIZE,
            test_size=self.params.TEST_SIZE,
            val_size=self.params.VAL_SIZE,
            seed=self.params.SEED
        )
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        params = self.params

        return ModelTrainingConfig(
        root_dir=Path(config.root_dir),
        trained_model_path=Path(config.trained_model_path),
        tensorboard_log_dir=Path(config.tensorboard_log_dir),
        epochs=params.EPOCHS,
        learning_rate=params.LEARNING_RATE,
        batch_size=params.BATCH_SIZE,
        img_height=params.IMG_HEIGHT,
        img_width=params.IMG_WIDTH,
    )



    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.model_evaluation

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            metric_file_name=Path(config.metric_file_name),
            mlflow_uri=config.mlflow_uri,
            batch_size=params.batch_size,
            img_height=params.img_height,
            img_width=params.img_width,
            target_metric=params.target_metric
        )