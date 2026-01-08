"""
CNN Model Builder
Xây dựng các kiến trúc CNN: ResNet50, MobileNetV2, Custom CNN
"""
import logging
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras import layers, models

from ..config import CNN_MODEL_TYPE, IMAGE_SIZE, CLASSES, CNN_UNFREEZE_LAYERS

logger = logging.getLogger(__name__)


class ModelBuilder:
    """
    Class xây dựng các CNN models
    """
    
    @staticmethod
    def build_resnet50(num_classes=None, input_shape=None, trainable_layers=CNN_UNFREEZE_LAYERS):
        """
        Xây dựng ResNet50 với transfer learning
        
        Args:
            num_classes: Số classes (default từ config)
            input_shape: Shape đầu vào (default từ config)
            trainable_layers: Số layers cuối được train
            
        Returns:
            Model Keras
        """
        if num_classes is None:
            num_classes = len(CLASSES)
        
        if input_shape is None:
            input_shape = (*IMAGE_SIZE, 3)
        
        logger.info(f"Building ResNet50: {num_classes} classes, input={input_shape}")
        
        # Load pretrained ResNet50 (ImageNet weights)
        base_model = ResNet50(
            include_top=False,         # Không dùng FC layers của ImageNet
            weights='imagenet',        # Pretrained weights
            input_shape=input_shape
        )
        
        # Freeze base model ban đầu
        base_model.trainable = False
        
        logger.info(f"ResNet50 base loaded: {len(base_model.layers)} layers (frozen)")
        
        # Build custom head cho classification
        model = models.Sequential([
            base_model,
            
            # Global pooling thay vì flatten → giảm params
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu', name='fc1'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu', name='fc2'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax', name='predictions')
        ], name='ResNet50_ImageRecognition')
        
        logger.info(f"Model built: {model.count_params():,} total params")
        
        return model, base_model
    
    @staticmethod
    def unfreeze_base_model(base_model, num_layers=CNN_UNFREEZE_LAYERS):
        """
        Unfreeze một số layers cuối của base model để fine-tune
        
        Args:
            base_model: Base model (ResNet50/MobileNetV2)
            num_layers: Số layers cuối được unfreeze
        """
        base_model.trainable = True
        
        # Freeze tất cả trừ num_layers cuối
        total_layers = len(base_model.layers)
        freeze_until = total_layers - num_layers
        
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        
        for layer in base_model.layers[freeze_until:]:
            layer.trainable = True
        
        trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
        
        logger.info(f"Fine-tuning enabled: {trainable_count}/{total_layers} layers trainable")
        
        return base_model
    
    @staticmethod
    def build_mobilenet_v2(num_classes=None, input_shape=None):
        """
        Xây dựng MobileNetV2 - lightweight alternative
        
        Args:
            num_classes: Số classes
            input_shape: Shape đầu vào
            
        Returns:
            Model Keras
        """
        if num_classes is None:
            num_classes = len(CLASSES)
        
        if input_shape is None:
            input_shape = (*IMAGE_SIZE, 3)
        
        logger.info(f"Building MobileNetV2: {num_classes} classes")
        
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ], name='MobileNetV2_ImageRecognition')
        
        logger.info(f"MobileNetV2 built: {model.count_params():,} params")
        
        return model, base_model
    
    @staticmethod
    def build_custom_cnn(num_classes=None, input_shape=None):
        """
        Xây dựng Custom CNN from scratch
        
        Args:
            num_classes: Số classes
            input_shape: Shape đầu vào
            
        Returns:
            Model Keras
        """
        if num_classes is None:
            num_classes = len(CLASSES)
        
        if input_shape is None:
            input_shape = (*IMAGE_SIZE, 3)
        
        logger.info(f"Building Custom CNN: {num_classes} classes")
        
        model = models.Sequential([
            # Input
            layers.Input(shape=input_shape),
            
            # Conv Block 1
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ], name='Custom_CNN')
        
        logger.info(f"Custom CNN built: {model.count_params():,} params")
        
        return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test build models
    builder = ModelBuilder()
    
    print("\n=== Testing ResNet50 ===")
    model, base = builder.build_resnet50()
    model.summary()
    
    print("\n=== Testing MobileNetV2 ===")
    model2, base2 = builder.build_mobilenet_v2()
    model2.summary()
