from src.BloodCellClassifier import logger
from src.BloodCellClassifier.pipeline.Data_Ingestion_Pipeline import DataIngestionTrainingPipeline
# from src.BloodCellClassifier.pipeline.Data_Preprocessing_Pipeline import DataPreprocessingTrainingPipeline
# from src.BloodCellClassifier.pipeline.Model_Training_Pipeline import ModelTrainingPipeline
# from src.BloodCellClassifier.pipeline.Model_Evaluation_Pipeline import ModelEvaluationPipeline

# dagshub.init(repo_owner='gowtham-dd', repo_name='Introvert-Vs-Extrovert', mlflow=True)


STAGE_NAME="Data Ingestion stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e
