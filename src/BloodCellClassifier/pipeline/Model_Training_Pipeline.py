
from BloodCellClassifier.config.configuration import ConfigurationManager
from BloodCellClassifier.components.Model_Training import ModelTrainer
from BloodCellClassifier import logger
import os
from tensorflow.keras.models import Sequential, load_model  # Make sure load_model is imported


STAGE_NAME = "Data Ingestion stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self, train_gen=None, val_gen=None, test_gen=None):
        try:
            config = ConfigurationManager()
            model_training_config = config.get_model_training_config()
            model_trainer = ModelTrainer(config=model_training_config)
            
            # Skip entire training stage if model exists
            if model_trainer.should_skip_training():
                logger.info("Model already exists. Skipping training stage.")
                trained_model = load_model(model_training_config.trained_model_path)
            else:
                # Train the model (using the generators from transformation)
                trained_model, history = model_trainer.train(train_gen, val_gen)
                
                # Save the model
                model_trainer.save_model(trained_model)
            
            # Evaluate on test set if model exists
            if trained_model and test_gen:
                test_loss, test_acc = trained_model.evaluate(test_gen)
                logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
            
            return trained_model

        except Exception as e:
            logger.exception(f"Error in training pipeline: {e}")
            raise e
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e