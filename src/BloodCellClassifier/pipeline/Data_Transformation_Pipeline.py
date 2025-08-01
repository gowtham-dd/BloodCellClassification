from BloodCellClassifier.config.configuration import ConfigurationManager
from BloodCellClassifier.components.Data_Transformation import DataTransformation
from BloodCellClassifier import logger
import os
STAGE_NAME = "Data Ingestion stage"


class DataIngestionTransformationPipeline:
    def __init__(self):
        pass

    def main(self):

        try:
    # Initialize configuration
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
    
    # Initialize data transformation
            data_transformation = DataTransformation(config=data_transformation_config)
    
    # Step 1: Create dataframe from image directories
            logger.info("Creating dataframe from image directories...")
            bloodCell_df = data_transformation.create_dataframe()
    
    # Step 2: Split data into train, validation, test sets
            logger.info("Splitting data into train/val/test sets...")
            train_set, val_set, test_images = data_transformation.split_data(bloodCell_df)
    
    # Step 3: Create data generators
            logger.info("Creating data generators...")
            train_gen, val_gen, test_gen = data_transformation.get_data_generators(
        train_set, val_set, test_images
    )
    
    # Optional: Save the split datasets
            logger.info("Saving split datasets...")
            os.makedirs(data_transformation_config.root_dir, exist_ok=True)
            train_set.to_csv(os.path.join(data_transformation_config.root_dir, "train_set.csv"), index=False)
            val_set.to_csv(os.path.join(data_transformation_config.root_dir, "val_set.csv"), index=False)
            test_images.to_csv(os.path.join(data_transformation_config.root_dir, "test_images.csv"), index=False)
    
            logger.info("Data transformation completed successfully!")

        except Exception as e:
            logger.exception(f"Error in data transformation pipeline: {e}")
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTransformationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e