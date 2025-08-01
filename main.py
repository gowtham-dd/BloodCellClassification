from src.BloodCellClassifier import logger
from src.BloodCellClassifier.pipeline.Data_Ingestion_Pipeline import DataIngestionTrainingPipeline
from src.BloodCellClassifier.pipeline.Data_Transformation_Pipeline import DataIngestionTransformationPipeline
from src.BloodCellClassifier.pipeline.Model_Training_Pipeline import ModelTrainingPipeline
from src.BloodCellClassifier.pipeline.Model_Evaluation_Pipeline import ModelEvaluationPipeline

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



STAGE_NAME="Data Transformation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataIngestionTransformationPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME="Model Training stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e





STAGE_NAME="Model Evaluation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e
