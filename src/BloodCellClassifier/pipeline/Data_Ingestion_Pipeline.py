from BloodCellClassifier.config.configuration import ConfigurationManager
from BloodCellClassifier.components.Data_Ingestion import DataIngestion
from BloodCellClassifier import logger
import os
STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)

    # Only download and extract if file doesn't exist
            if not os.path.exists(data_ingestion_config.local_data_file):
                data_ingestion.download_file()
                data_ingestion.extract_zip_file()
            else:
                logger.info(
                f"File already exists at {data_ingestion_config.local_data_file}. Skipping download and extraction."
                )
        except Exception as e:
            raise e




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e