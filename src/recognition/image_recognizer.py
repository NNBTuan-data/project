"""
Module nhận diện ảnh bằng similarity search
Chuyển đổi từ recognize.py
"""
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple

from ..config import (COMBINED_FEATURES_FILE, LABELS_FILE, SIMILARITY_THRESHOLD, 
                      TOP_K_RESULTS, IMAGE_SIZE)
from ..preprocessing.image_processor import ImageProcessor
from ..features.lbp_extractor import LBPExtractor  
from ..features.hog_extractor import HOGExtractor
from .similarity_calculator import SimilarityCalculator

logger = logging.getLogger(__name__)


class ImageRecognizer:
    """
    Class nhận diện ảnh bằng similarity search
    Tìm top-K ảnh tương đồng nhất từ database
    """
    
    def __init__(self, features_path: Path = None, labels_path: Path = None):
        """
        Khởi tạo ImageRecognizer
        
        Args:
            features_path: Đường dẫn file features
            labels_path: Đường dẫn file labels
        """
        self.features_path = features_path or COMBINED_FEATURES_FILE
        self.labels_path = labels_path or LABELS_FILE
        
        # Khởi tạo components
        self.image_processor = ImageProcessor(target_size=IMAGE_SIZE)
        self.lbp_extractor = LBPExtractor()
        self.hog_extractor = HOGExtractor()
        self.similarity_calc = SimilarityCalculator()
        
        # Load database
        self.features = None
        self.labels = None
        self._load_database()
        
        logger.info(f"Khởi tạo ImageRecognizer: {len(self.features)} ảnh trong database")
    
    def _load_database(self) -> None:
        """Load features và labels từ file"""
        try:
            self.features = np.load(str(self.features_path))
            self.labels = np.load(str(self.labels_path))
            logger.info(f"Đã load database: features{self.features.shape}, labels{self.labels.shape}")
        except Exception as e:
            logger.error(f"Không thể load database: {e}")
            raise
    
    def extract_query_features(self, img_path: str) -> np.ndarray:
        """
        Trích xuất đặc trưng từ ảnh query
        
        Args:
            img_path: Đường dẫn ảnh query
            
        Returns:
            Feature vector
        """
        # Đọc và tiền xử lý ảnh
        img = self.image_processor.read_image(img_path, size=IMAGE_SIZE)
        
        # Trích xuất LBP + HOG (combined)
        lbp_feat = self.lbp_extractor.extract(img)
        hog_feat = self.hog_extractor.extract(img)
        
        # Kết hợp features
        features = np.concatenate([lbp_feat, hog_feat])
        
        # Chuẩn hóa L2
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        logger.debug(f"Extracted query features: shape={features.shape}")
        return features
    
    def recognize(self, img_path: str, topk: int = TOP_K_RESULTS, 
                 threshold: float = SIMILARITY_THRESHOLD) -> Tuple[List, List]:
        """
        Nhận diện ảnh và trả về top-K kết quả
        
        Args:
            img_path: Đường dẫn ảnh cần nhận diện
            topk: Số kết quả top trả về
            threshold: Ngưỡng similarity tối thiểu
            
        Returns:
            (results, match_indices)
            results: [(label, similarity, index), ...]
            match_indices: [idx1, idx2, ...]
        """
        logger.info(f"Nhận diện ảnh: {img_path}")
        
        # Trích xuất đặc trưng query
        qv = self.extract_query_features(img_path)
        
        # Tính cosine similarity với tất cả ảnh trong database
        sims = self.similarity_calc.cosine_similarity(qv, self.features)
        
        # Sắp xếp và lấy top-K
        sorted_idx = np.argsort(sims)[::-1][:topk]
        
        # Tạo kết quả
        results = []
        match_indices = []
        
        for i, idx in enumerate(sorted_idx):
            sim = float(sims[idx])
            label = int(self.labels[idx])
            
            # Nếu top-1 có similarity < threshold → "Không nhận diện được"
            if i == 0 and sim < threshold:
                label_str = "Unknown"
                logger.warning(f"Top-1 similarity {sim:.3f} < threshold {threshold}")
            else:
                label_str = str(label)
            
            results.append((label_str, sim, int(idx)))
            match_indices.append(int(idx))
        
        logger.info(f"Top-1: label={results[0][0]}, similarity={results[0][1]:.3f}")
        
        return results, match_indices


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    recognizer = ImageRecognizer()
    print(f"Database size: {len(recognizer.features)} images")
