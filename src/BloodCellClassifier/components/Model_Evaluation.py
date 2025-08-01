import os
import mlflow
import mlflow.tensorflow
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from urllib.parse import urlparse
from pathlib import Path
from BloodCellClassifier.utils.common import save_json
from BloodCellClassifier import logger
from src.BloodCellClassifier.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config):
        self.config = config
        self.image_size = (self.config.img_height, self.config.img_width)

    def _load_test_generator(self):
        """Create test data generator from saved CSV"""
        try:
            test_df = pd.read_csv(self.config.test_data_path)
            
            datagen = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
            )
            
            test_gen = datagen.flow_from_dataframe(
                dataframe=test_df,
                x_col="filepaths",
                y_col="labels",
                target_size=self.image_size,
                batch_size=self.config.batch_size,
                class_mode="categorical",
                shuffle=False
            )
            return test_gen
        except Exception as e:
            logger.error(f"Error creating test generator: {e}")
            raise

    def _load_model(self):
        """Load the saved TensorFlow model"""
        try:
            return tf.keras.models.load_model(self.config.model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def evaluate_model(self):
        """Evaluate model and return metrics"""
        try:
            model = self._load_model()
            test_gen = self._load_test_generator()

            # Evaluate model
            loss, accuracy = model.evaluate(test_gen)
            
            metrics = {
                "loss": float(loss),
                "accuracy": float(accuracy)
            }

            # Save metrics
            save_json(Path(self.config.metric_file_name), metrics)
            logger.info(f"Evaluation metrics: {metrics}")
            return model, metrics

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    def log_into_mlflow(self):
        """Log evaluation results to MLflow"""
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        try:
            with mlflow.start_run():
                # Load model and evaluate
                model, metrics = self.evaluate_model()

                # Log parameters
                mlflow.log_params({
                    "batch_size": self.config.batch_size,
                    "img_height": self.config.img_height,
                    "img_width": self.config.img_width
                })

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log model
                if tracking_url_type_store != "file":
                    mlflow.tensorflow.log_model(
                        model,
                        artifact_path="blood_cell_model",
                        registered_model_name="BloodCell_CNN_Model"
                    )
                else:
                    mlflow.log_artifacts(str(self.config.model_path), artifact_path="blood_cell_model")

                logger.info("Model evaluation and MLflow logging completed successfully")

        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")
            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
            raise