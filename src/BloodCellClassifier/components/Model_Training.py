import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from BloodCellClassifier import logger
from src.BloodCellClassifier.entity.config_entity import ModelTrainingConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        # Create directories if they don't exist
        os.makedirs(self.config.root_dir, exist_ok=True)
        os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)

    def _model_exists(self):
        """Check if model is already saved in TF2.12 format"""
        required_files = [
            os.path.join(self.config.trained_model_path, "saved_model.pb"),
            os.path.join(self.config.trained_model_path, "variables/variables.index")
        ]
        return all(os.path.exists(f) for f in required_files)

    def build_model(self):
        """Builds the CNN model (same as original code)"""
        model = Sequential([
            Conv2D(128, (8, 8), strides=(3, 3), activation='relu', input_shape=(224, 224, 3)),
            BatchNormalization(),
            
            Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding="same"),
            BatchNormalization(),
            MaxPooling2D((3, 3)),
            
            Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"),
            BatchNormalization(),
            Conv2D(256, (1, 1), strides=(1, 1), activation='relu', padding="same"),
            BatchNormalization(),
            Conv2D(256, (1, 1), strides=(1, 1), activation='relu', padding="same"),
            BatchNormalization(),
            
            Conv2D(512, (3, 3), activation='relu', padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(512, (3, 3), activation='relu', padding="same"),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu', padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(512, (3, 3), activation='relu', padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='softmax')
        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(learning_rate=self.config.learning_rate),
            metrics=['accuracy']
        )
        return model

    def should_skip_training(self):
        """Check if training should be skipped (model already exists)"""
        return self._model_exists()

    def train(self, train_gen, val_gen):
        """Trains only if model doesn't exist"""
        if self.should_skip_training():
            logger.info(f"Model already exists at {self.config.trained_model_path}. Skipping training.")
            return load_model(self.config.trained_model_path), None
        
        model = self.build_model()
        
        callbacks = [
            TensorBoard(log_dir=self.config.tensorboard_log_dir),
            EarlyStopping(patience=3, restore_best_weights=True),
            LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
        ]
        
        history = model.fit(
            train_gen,
            epochs=self.config.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        return model, history

    def save_model(self, model):
        """Saves in TF2.12 format (creates .pb + assets/ + variables/)"""
        if self._model_exists():
            logger.info("Model already exists. Skipping save.")
            return
            
        model.save(
            self.config.trained_model_path,
            save_format="tf"  # This creates the full TF2.12 structure
        )
        logger.info(f"Model saved in TF2.12 format at: {self.config.trained_model_path}")
        logger.info("Contains: saved_model.pb, assets/, variables/")