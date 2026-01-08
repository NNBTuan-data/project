"""
Data Augmentation cho CNN Training
Tạo ImageDataGenerator với augmentation
"""
import logging
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ..config import (DATASET_DIR, IMAGE_SIZE, CNN_AUGMENTATION, 
                      CNN_BATCH_SIZE, CNN_VALIDATION_SPLIT)

logger = logging.getLogger(__name__)


class DataAugmenter:
    """
    Class quản lý data augmentation cho CNN training
    """
    
    def __init__(self, augmentation_params=None, validation_split=CNN_VALIDATION_SPLIT):
        """
        Khởi tạo DataAugmenter
        
        Args:
            augmentation_params: Dict augmentation params (default từ config)
            validation_split: Tỷ lệ validation (default 0.2)
        """
        self.augmentation_params = augmentation_params or CNN_AUGMENTATION
        self.validation_split = validation_split
        
        # Training data generator (with augmentation)
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values to [0,1]
            validation_split=self.validation_split,
            **self.augmentation_params
        )
        
        # Validation data generator (only rescale, no augmentation)
        self.val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split
        )
        
        logger.info(f"DataAugmenter initialized: validation_split={validation_split}")
        logger.info(f"Augmentation: {self.augmentation_params}")
    
    def create_generators(self, dataset_dir=None, batch_size=CNN_BATCH_SIZE, 
                         target_size=None, shuffle_train=True):
        """
        Tạo train và validation generators từ directory
        
        Args:
            dataset_dir: Thư mục dataset (default từ config)
            batch_size: Batch size
            target_size: Kích thước ảnh (default từ config)
            shuffle_train: Shuffle training data
            
        Returns:
            (train_generator, val_generator)
        """
        if dataset_dir is None:
            dataset_dir = DATASET_DIR
        
        if target_size is None:
            target_size = IMAGE_SIZE
        
        logger.info(f"Creating generators from {dataset_dir}")
        logger.info(f"Target size: {target_size}, Batch size: {batch_size}")
        
        # Training generator (with augmentation)
        train_generator = self.train_datagen.flow_from_directory(
            str(dataset_dir),
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=shuffle_train,
            seed=42
        )
        
        # Validation generator (no augmentation)
        val_generator = self.val_datagen.flow_from_directory(
            str(dataset_dir),
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        logger.info(f"Train samples: {train_generator.samples}")
        logger.info(f"Validation samples: {val_generator.samples}")
        logger.info(f"Classes: {train_generator.class_indices}")
        
        return train_generator, val_generator
    
    def get_steps_per_epoch(self, generator):
        """Tính số steps per epoch"""
        return len(generator)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    augmenter = DataAugmenter()
    train_gen, val_gen = augmenter.create_generators()
    
    print(f"\nTrain steps: {len(train_gen)}")
    print(f"Val steps: {len(val_gen)}")
    print(f"Classes: {train_gen.class_indices}")
