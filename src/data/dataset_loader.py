import os
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ..config import DATASET_DIR, CLASSES
logger = logging.getLogger(__name__)
class DatasetLoader:
    def __init__(self, dataset_path: Path = None):
        self.dataset_path = dataset_path or DATASET_DIR
        self.label_encoder = LabelEncoder()
        logger.info(f"Khởi tạo DatasetLoader: {self.dataset_path}")
    def load_dataset(self, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')) -> Tuple[List[str], np.ndarray]:
        img_paths = []
        labels = []
        classes = sorted([d for d in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, d))])
        logger.info(f"Tìm thấy {len(classes)} classes: {classes}")
        for cls in classes:
            cls_path = os.path.join(self.dataset_path, cls)
            files = [f for f in os.listdir(cls_path) 
                    if f.lower().endswith(extensions)]
            files.sort(key=lambda x: x.lower())            
            logger.debug(f"Class '{cls}': {len(files)} ảnh")
            for f in files:
                img_paths.append(os.path.join(cls_path, f))
                labels.append(cls)
        labels_encoded = self.label_encoder.fit_transform(labels)
        logger.info(f"Đã load {len(img_paths)} ảnh từ {len(classes)} classes")
        logger.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        return img_paths, labels_encoded
    def get_class_names(self) -> List[str]:
        classes = sorted([d for d in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, d))])
        return classes
    def get_class_distribution(self) -> dict:
        distribution = {}
        classes = self.get_class_names()
        for cls in classes:
            cls_path = os.path.join(self.dataset_path, cls)
            count = len([f for f in os.listdir(cls_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            distribution[cls] = count
        logger.info(f"Distribution: {distribution}")
        return distribution
    def decode_labels(self, labels_encoded: np.ndarray) -> List[str]:
        if not hasattr(self.label_encoder, 'classes_'):
            # Nếu chưa fit, load dataset để fit
            logger.warning("LabelEncoder chưa fit, đang load dataset...")
            self.load_dataset()
        return list(self.label_encoder.inverse_transform(labels_encoded))
if __name__ == "__main__":
    # Test DatasetLoader
    logging.basicConfig(level=logging.DEBUG)
    loader = DatasetLoader()
    paths, labels = loader.load_dataset()
    print(f"\nTổng số ảnh: {len(paths)}")
    print(f"Số classes: {len(loader.get_class_names())}")
    print(f"Classes: {loader.get_class_names()}")
    print(f"\nDistribution:")
    for cls, count in loader.get_class_distribution().items():
        print(f"  {cls}: {count}")
