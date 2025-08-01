



from BloodCellClassifier.config.configuration import ConfigurationManager
from BloodCellClassifier.components.Model_Evaluation import ModelEvaluation
from BloodCellClassifier import logger
import os
STAGE_NAME = "Data Ingestion stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):

        try:
            config = ConfigurationManager()
            eval_config = config.get_model_evaluation_config()
            evaluator = ModelEvaluation(config=eval_config)
            evaluator.log_into_mlflow()
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e