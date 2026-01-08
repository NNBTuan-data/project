import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, 
                                       ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.optimizers import Adam

from .model_builder import ModelBuilder
from ..preprocessing.data_augmenter import DataAugmenter
from ..config import (CNN_MODEL_TYPE, CNN_MODEL_PATH, CNN_EPOCHS, CNN_BATCH_SIZE,
                      CNN_LEARNING_RATE, CNN_EARLY_STOPPING_PATIENCE,
                      CNN_REDUCE_LR_PATIENCE, CNN_REDUCE_LR_FACTOR,
                      CNN_UNFREEZE_LAYERS, MODEL_DIR, CLASSES)

logger = logging.getLogger(__name__)


class CNNTrainer:
    """
    Class training CNN models
    """
    
    def __init__(self, model_type=None):
        """
        Khởi tạo CNNTrainer
        
        Args:
            model_type: Loại model ('resnet50', 'mobilenet', 'custom')
        """
        self.model_type = model_type or CNN_MODEL_TYPE
        self.model = None
        self.base_model = None
        self.history = None
        
        logger.info(f"CNNTrainer initialized: model_type={self.model_type}")
    
    def build_model(self):
        """Xây dựng model architecture"""
        logger.info(f"Building {self.model_type} model...")
        
        builder = ModelBuilder()
        
        if self.model_type == 'resnet50':
            self.model, self.base_model = builder.build_resnet50()
        elif self.model_type == 'mobilenet':
            self.model, self.base_model = builder.build_mobilenet_v2()
        elif self.model_type == 'custom':
            self.model = builder.build_custom_cnn()
            self.base_model = None
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        logger.info(f"Model built: {self.model.count_params():,} parameters")
        
        return self.model
    
    def compile_model(self, learning_rate=CNN_LEARNING_RATE):
        """
        Compile model với optimizer và loss
        
        Args:
            learning_rate: Learning rate
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        logger.info(f"Model compiled: lr={learning_rate}")
    
    def create_callbacks(self, model_path=CNN_MODEL_PATH):
        """
        Tạo callbacks cho training
        
        Args:
            model_path: Đường dẫn lưu model
            
        Returns:
            List callbacks
        """
        callbacks = [
            # Save best model
            ModelCheckpoint(
                str(model_path),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=CNN_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=CNN_REDUCE_LR_FACTOR,
                patience=CNN_REDUCE_LR_PATIENCE,
                verbose=1,
                min_lr=1e-7
            ),
            
            # TensorBoard (optional)
            # TensorBoard(log_dir=str(MODEL_DIR / 'logs'))
        ]
        
        logger.info(f"Created {len(callbacks)} callbacks")
        return callbacks
    
    def train(self, epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE, 
             fine_tune=True, fine_tune_epochs=10):
        """
        Train model với 2 phases: freeze + fine-tune
        
        Args:
            epochs: Số epochs phase 1 (freeze base)
            batch_size: Batch size
            fine_tune: Có fine-tune base model không
            fine_tune_epochs: Số epochs phase 2 (fine-tune)
            
        Returns:
            History object
        """
        logger.info("="*70)
        logger.info("BẮT ĐẦU TRAINING CNN")
        logger.info("="*70)
        
        # Prepare data
        augmenter = DataAugmenter()
        train_gen, val_gen = augmenter.create_generators(batch_size=batch_size)
        
        # Phase 1: Train with frozen base
        logger.info(f"\n--- PHASE 1: Training top layers ({epochs} epochs) ---")
        
        callbacks = self.create_callbacks()
        
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        phase1_acc = max(self.history.history['val_accuracy'])
        logger.info(f"Phase 1 best accuracy: {phase1_acc*100:.2f}%")
        
        # Phase 2: Fine-tune (if requested)
        if fine_tune and self.base_model is not None:
            logger.info(f"\n--- PHASE 2: Fine-tuning ({fine_tune_epochs} epochs) ---")
            
            # Unfreeze base model
            builder = ModelBuilder()
            self.base_model = builder.unfreeze_base_model(
                self.base_model, 
                num_layers=CNN_UNFREEZE_LAYERS
            )
            
            # Recompile with lower learning rate
            self.compile_model(learning_rate=CNN_LEARNING_RATE / 10)
            
            # Continue training
            history2 = self.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=fine_tune_epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Merge histories
            for key in self.history.history:
                self.history.history[key].extend(history2.history[key])
            
            phase2_acc = max(history2.history['val_accuracy'])
            logger.info(f"Phase 2 best accuracy: {phase2_acc*100:.2f}%")
        
        # Final results
        best_acc = max(self.history.history['val_accuracy'])
        best_epoch = self.history.history['val_accuracy'].index(best_acc) + 1
        
        logger.info("="*70)
        logger.info(f"✅ TRAINING COMPLETED!")
        logger.info(f"Best Validation Accuracy: {best_acc*100:.2f}% (epoch {best_epoch})")
        logger.info(f"Model saved to: {CNN_MODEL_PATH}")
        logger.info("="*70)
        
        return self.history
    
    def load_model(self, model_path=CNN_MODEL_PATH):
        """Load model đã train"""
        from tensorflow.keras.models import load_model
        
        self.model = load_model(str(model_path))
        logger.info(f"Model loaded from {model_path}")
        
        return self.model
    
    def evaluate(self, test_data):
        """Evaluate model trên test set"""
        results = self.model.evaluate(test_data, verbose=1)
        
        logger.info(f"Test Loss: {results[0]:.4f}")
        logger.info(f"Test Accuracy: {results[1]*100:.2f}%")
        
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    trainer = CNNTrainer(model_type='resnet50')
    trainer.build_model()
    trainer.compile_model()
    
    print("\nModel ready for training!")
    print("Run: trainer.train()")
