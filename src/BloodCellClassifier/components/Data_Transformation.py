from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import numpy as np
from BloodCellClassifier import logger
from src.BloodCellClassifier.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.class_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
        os.makedirs(self.config.root_dir, exist_ok=True)
        
    def _output_files_exist(self):
        """Check if transformation output files already exist"""
        required_files = [
            os.path.join(self.config.root_dir, "train_set.csv"),
            os.path.join(self.config.root_dir, "val_set.csv"),
            os.path.join(self.config.root_dir, "test_set.csv")
        ]
        return all(os.path.exists(f) for f in required_files)

    def create_dataframe(self):
        try:
            filepaths = []
            labels = []
            
            # Map directory lists to their class labels
            dir_class_map = {
                'EOSINOPHIL': self.config.EOSINOPHIL_dirs,
                'LYMPHOCYTE': self.config.LYMPHOCYTE_dir,
                'MONOCYTE': self.config.MONOCYTE_dirs,
                'NEUTROPHIL': self.config.NEUTROPHIL_dirs
            }
            
            for class_name, dir_list in dir_class_map.items():
                for dir_path in dir_list:
                    for f in os.listdir(dir_path):
                        fpath = os.path.join(dir_path, f)
                        filepaths.append(fpath)
                        labels.append(class_name)
            
            # Create dataframe
            Fseries = pd.Series(filepaths, name="filepaths")
            Lseries = pd.Series(labels, name="labels")
            bloodCell_df = pd.concat([Fseries, Lseries], axis=1)
            
            logger.info(f"Created dataframe with {len(bloodCell_df)} samples")
            logger.info("Class distribution:\n" + str(bloodCell_df["labels"].value_counts()))
            
            return bloodCell_df
            
        except Exception as e:
            logger.error(f"Error creating dataframe: {e}")
            raise e

    def split_data(self, df):
        try:
            # Split data
            train_images, test_images = train_test_split(
                df, 
                test_size=self.config.test_size, 
                random_state=self.config.seed
            )
            train_set, val_set = train_test_split(
                train_images, 
                test_size=self.config.val_size, 
                random_state=self.config.seed
            )
            
            logger.info(f"Train set size: {len(train_set)}")
            logger.info(f"Validation set size: {len(val_set)}")
            logger.info(f"Test set size: {len(test_images)}")
            
            return train_set, val_set, test_images
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise e

    def get_data_generators(self, train_set=None, val_set=None, test_set=None):
        """Get data generators, loading from files if not provided"""
        try:
            # Load data from files if not provided
            if train_set is None or val_set is None or test_set is None:
                logger.info("Loading data from saved CSV files")
                train_set = pd.read_csv(os.path.join(self.config.root_dir, "train_set.csv"))
                val_set = pd.read_csv(os.path.join(self.config.root_dir, "val_set.csv"))
                test_set = pd.read_csv(os.path.join(self.config.root_dir, "test_set.csv"))
            
            # Create image generator
            image_gen = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
            )
            
            train_gen = image_gen.flow_from_dataframe(
                dataframe=train_set,
                x_col="filepaths",
                y_col="labels",
                target_size=(self.config.img_height, self.config.img_width),
                color_mode='rgb',
                class_mode="categorical",
                batch_size=self.config.batch_size,
                shuffle=False
            )
            
            test_gen = image_gen.flow_from_dataframe(
                dataframe=test_set,
                x_col="filepaths",
                y_col="labels",
                target_size=(self.config.img_height, self.config.img_width),
                color_mode='rgb',
                class_mode="categorical",
                batch_size=self.config.batch_size,
                shuffle=False
            )
            
            val_gen = image_gen.flow_from_dataframe(
                dataframe=val_set,
                x_col="filepaths",
                y_col="labels",
                target_size=(self.config.img_height, self.config.img_width),
                color_mode='rgb',
                class_mode="categorical",
                batch_size=self.config.batch_size,
                shuffle=False
            )
            
            logger.info("Created data generators successfully")
            logger.info(f"Classes: {train_gen.class_indices}")
            
            return train_gen, val_gen, test_gen
            
        except Exception as e:
            logger.error(f"Error creating data generators: {e}")
            raise e

    def transform(self):
        """Main transformation method that checks for existing files"""
        try:
            if self._output_files_exist():
                logger.info("Transformation files already exist. Skipping data transformation.")
                return None, None, None
            else:
                logger.info("Starting data transformation")
                bloodCell_df = self.create_dataframe()
                train_set, val_set, test_set = self.split_data(bloodCell_df)
                
                # Save the dataframes
                train_set.to_csv(os.path.join(self.config.root_dir, "train_set.csv"), index=False)
                val_set.to_csv(os.path.join(self.config.root_dir, "val_set.csv"), index=False)
                test_set.to_csv(os.path.join(self.config.root_dir, "test_set.csv"), index=False)
                
                logger.info("Data transformation completed successfully!")
                return train_set, val_set, test_set
                
        except Exception as e:
            logger.exception(f"Error in data transformation pipeline: {e}")
            raise e