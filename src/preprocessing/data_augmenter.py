"""
Data Augmentation THỦ CÔNG (Manual Implementation)
KHÔNG dùng ImageDataGenerator của Keras
Sử dụng: OpenCV, NumPy
"""
import logging
import cv2
import numpy as np
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.utils import Sequence

from ..config import (DATASET_DIR, IMAGE_SIZE, CNN_BATCH_SIZE, 
                      CNN_VALIDATION_SPLIT, CLASSES)

logger = logging.getLogger(__name__)

class ManualAugmenter:
    @staticmethod
    def rotate(image: np.ndarray) -> np.ndarray:
        angle = random.uniform(-30, 30)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), 
                            flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_REFLECT_101)

    @staticmethod
    def shift(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        tx = random.uniform(-0.2, 0.2) * w 
        ty = random.uniform(-0.2, 0.2) * h
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # 2. Áp dụng biến đổi
        return cv2.warpAffine(image, M, (w, h), 
                            borderMode=cv2.BORDER_REFLECT_101)

    @staticmethod
    def flip(image: np.ndarray) -> np.ndarray:
        """Lật ảnh ngang (Horizontal Flip)"""
        # Encode: 1=Flip ngang, 0=Flip dọc, -1=Cả hai
        return cv2.flip(image, 1)

    @staticmethod
    def augment(image: np.ndarray) -> np.ndarray:
        """Áp dụng ngẫu nhiên các phép biến đổi"""
        augmented = image.copy()
        
        # 50% cơ hội xoay ảnh
        if random.random() < 0.5:
            augmented = ManualAugmenter.rotate(augmented)
            
        # 50% cơ hội lật ảnh
        if random.random() < 0.5:
            augmented = ManualAugmenter.flip(augmented)
            
        # 30% cơ hội dịch chuyển
        if random.random() < 0.3:
            augmented = ManualAugmenter.shift(augmented)
            
        return augmented


class CustomDataGenerator(Sequence):
    """
    Generator dữ liệu tự xây dựng (Custom)
    Kế thừa từ Sequence để đưa vào model.fit()
    """
    def __init__(self, image_paths, labels, batch_size, target_size, num_classes, 
                 shuffle=True, is_training=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.is_training = is_training # True thì mới augment
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        """Số lượng batch trong 1 epoch"""
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """Lấy 1 batch dữ liệu"""
        # Lấy danh sách index cho batch này
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Lấy path và label tương ứng
        batch_paths = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]

        # Load ảnh và xử lý
        X, y = self.__data_generation(batch_paths, batch_labels)
        return X, y

    def on_epoch_end(self):
        """Sau mỗi epoch thì xáo trộn lại dữ liệu"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_paths, batch_labels):
        """Load và xử lý batch"""
        # Khởi tạo mảng chứa batch
        X = np.empty((self.batch_size, *self.target_size, 3))
        y = np.empty((self.batch_size), dtype=int)

        for i, path in enumerate(batch_paths):
            # 1. Đọc ảnh bằng OpenCV
            img = cv2.imread(str(path))
            
            # Xử lý trường hợp đọc lỗi
            if img is None:
                # Tạo ảnh đen nếu lỗi (fallback)
                img = np.zeros((*self.target_size, 3), dtype=np.uint8)
            else:
                # BGR -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize
                img = cv2.resize(img, self.target_size)

            # 2. Augmentation (Nếu là tập Train)
            if self.is_training:
                img = ManualAugmenter.augment(img)

            # 3. Normalize [0, 1]
            X[i,] = img / 255.0
            
            # 4. Label
            y[i] = batch_labels[i]

        # One-hot encoding thủ công cho labels
        # equivalent to tf.keras.utils.to_categorical
        y_one_hot = np.eye(self.num_classes)[y]
        
        return X, y_one_hot


class DataAugmenter:
    """Class wrapper để tương thích với code cũ"""
    def __init__(self, validation_split=CNN_VALIDATION_SPLIT):
        self.validation_split = validation_split
        self.logger = logger

    def create_generators(self, dataset_dir=None, batch_size=CNN_BATCH_SIZE, 
                         target_size=IMAGE_SIZE, shuffle_train=True):
        
        # 1. Load danh sách file thủ công
        dataset_dir = Path(dataset_dir or DATASET_DIR)
        image_paths = []
        labels = []
        
        # Quét thư mục để lấy class
        class_names = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
        class_map = {name: i for i, name in enumerate(class_names)}
        
        self.logger.info("Đang load danh sách file thủ công (Manual Loading)...")
        for cls_name in class_names:
            cls_dir = dataset_dir / cls_name
            # Lấy tất cả ảnh jpg, png, ...
            files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            for img_path in files:
                image_paths.append(str(img_path))
                labels.append(class_map[cls_name])
                
        # 2. Chia tập Train/Val thủ công
        # Zip để shuffle đồng bộ
        combined = list(zip(image_paths, labels))
        random.shuffle(combined) # Xáo trộn ngẫu nhiên
        image_paths[:], labels[:] = zip(*combined)
        
        # Cắt theo tỷ lệ validation
        split_idx = int(len(image_paths) * (1 - self.validation_split))
        
        X_train, y_train = image_paths[:split_idx], labels[:split_idx]
        X_val, y_val = image_paths[split_idx:], labels[split_idx:]
        
        self.logger.info(f"Manual Split: {len(X_train)} train, {len(X_val)} val")

        # 3. Tạo Generator
        train_gen = CustomDataGenerator(X_train, y_train, batch_size, target_size, 
                                      len(class_names), shuffle=True, is_training=True)
                                      
        val_gen = CustomDataGenerator(X_val, y_val, batch_size, target_size,
                                    len(class_names), shuffle=False, is_training=False)
        
        # Thêm thuộc tính để tương thích với code cũ của dataset_loader/trainer
        train_gen.class_indices = class_map
        val_gen.class_indices = class_map
        train_gen.samples = len(X_train)
        val_gen.samples = len(X_val)
        
        return train_gen, val_gen
    
    def get_steps_per_epoch(self, generator):
        return len(generator)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    aug = DataAugmenter()
    t, v = aug.create_generators()
    print(f"Generator created: {len(t)} batches train")
