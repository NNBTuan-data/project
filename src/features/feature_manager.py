import os
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
from ..config import (DATASET_DIR, FEATURES_DIR, COLOR_FEATURES_FILE,
                      LBP_FEATURES_FILE, HOG_FEATURES_FILE, LABELS_FILE,
                      COMBINED_FEATURES_FILE, IMAGE_SIZE, FEATURE_WEIGHTS)
from ..data.dataset_loader import DatasetLoader
from ..preprocessing.image_processor import ImageProcessor
from .color_extractor import ColorExtractor
from .lbp_extractor import LBPExtractor
from .hog_extractor import HOGExtractor
logger = logging.getLogger(__name__)
class FeatureManager:
    def __init__(self, dataset_path: Path = None, output_dir: Path = None):
        self.dataset_path = dataset_path or DATASET_DIR
        self.output_dir = output_dir or FEATURES_DIR
        # Tạo thư mục output nếu chưa tồn tại
        self.output_dir.mkdir(exist_ok=True, parents=True)
        # Khởi tạo các components
        self.dataset_loader = DatasetLoader(self.dataset_path)
        self.image_processor = ImageProcessor(target_size=IMAGE_SIZE)
        self.color_extractor = ColorExtractor()
        self.lbp_extractor = LBPExtractor()
        self.hog_extractor = HOGExtractor()
        logger.info(f"Khởi tạo FeatureManager: dataset={self.dataset_path}, output={self.output_dir}")
    def extract_all_features(self, show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Bước 1: Load dataset
        logger.info("Đang load dataset...")
        paths, labels_all = self.dataset_loader.load_dataset()
        logger.info(f"TRÍCH XUẤT ĐẶC TRƯNG TỪ {len(paths)} ẢNH")
        print(f"\n{'='*70}")
        print(f"TRÍCH XUẤT ĐẶC TRƯNG TỪ {len(paths)} ẢNH")
        print(f"{'='*70}\n")
        # Khởi tạo danh sách lưu features
        color_list = []
        lbp_list = []
        hog_list = []
        valid_indices = []  # Track which images were successfully processed
        # Determine actual feature dimensions from first successful image
        hog_feature_dim = None
        if show_progress:
            print("Đang trích xuất đặc trưng...")
        # Bước 2: Trích xuất features cho từng ảnh
        iterator = paths
        for i, path in enumerate(iterator):
            try:
                # 2.1. Đọc ảnh (grayscale + tiền xử lý)
                img_gray = self.image_processor.read_image(path)
                
                # 2.2. Trích xuất 3 loại đặc trưng
                color_feat = self.color_extractor.extract(path)
                lbp_feat = self.lbp_extractor.extract(img_gray)
                hog_feat = self.hog_extractor.extract(img_gray)
                
                # Set HOG dimension from first successful extraction
                if hog_feature_dim is None:
                    hog_feature_dim = hog_feat.shape[0]
                    logger.info(f"HOG feature dimension: {hog_feature_dim}")
                
                # 2.3. Thêm vào danh sách
                color_list.append(color_feat)
                lbp_list.append(lbp_feat)
                hog_list.append(hog_feat)
                valid_indices.append(i)  # Track successful extraction
                
                if (i + 1) % 100 == 0:
                    logger.debug(f"Đã xử lý {i + 1}/{len(paths)} ảnh")
                    
            except Exception as e:
                logger.warning(f"Lỗi xử lý ảnh {path}: {e}")
                # Skip corrupted images
                continue
        
        # Filter labels to match valid images only
        labels = labels_all[valid_indices]
        
        # Chuyển sang numpy arrays
        colors = np.array(color_list)
        lbp_features = np.array(lbp_list)
        hog_features = np.array(hog_list)
        
        logger.info(f"Hoàn tất trích xuất: {len(valid_indices)}/{len(paths)} ảnh thành công")
        logger.info(f"Features shape - Colors:{colors.shape}, LBP:{lbp_features.shape}, HOG:{hog_features.shape}")
        
        # Bước 3: Lưu features
        self.save_features(colors, lbp_features, hog_features, labels)
        
        return colors, lbp_features, hog_features, labels
    
    def save_features(self, colors: np.ndarray, lbp_features: np.ndarray, 
                     hog_features: np.ndarray, labels: np.ndarray) -> None:
        """
        Lưu features ra file .npy
        
        Args:
            colors: Color features
            lbp_features: LBP features
            hog_features: HOG features
            labels: Labels
        """
        logger.info("Đang lưu features...")
        print(f"\nĐang lưu features vào {self.output_dir}...")
        
        # Lưu từng loại features riêng
        self._safe_save(colors, COLOR_FEATURES_FILE, "colors.npy")
        self._safe_save(lbp_features, LBP_FEATURES_FILE, "lbp.npy")
        self._safe_save(hog_features, HOG_FEATURES_FILE, "hog.npy")
        self._safe_save(labels, LABELS_FILE, "labels.npy")
        
        # Tạo combined features (optional)
        combined = self.combine_features(colors, lbp_features, hog_features)
        self._safe_save(combined, COMBINED_FEATURES_FILE, "features.npy (combined)")
        
        print(f"\n{'='*70}")
        print("✅ HOÀN TẤT! Kiểm tra tại: features/")
        print(f"{'='*70}\n")
    
    def _safe_save(self, data: np.ndarray, path: Path, name: str) -> bool:
        """
        Lưu numpy array an toàn
        
        Args:
            data: Dữ liệu cần lưu
            path: Đường dẫn lưu
            name: Tên file (để log)
            
        Returns:
            True nếu lưu thành công
        """
        try:
            np.save(str(path), data)
            size_mb = data.nbytes / 1e6
            logger.info(f"[OK] Lưu thành công: {name} ({size_mb:.1f} MB)")
            print(f"  [OK] {name} - {size_mb:.1f} MB - Shape: {data.shape}")
            return True
        except Exception as e:
            logger.error(f"[LỖI] Lưu thất bại {name}: {e}")
            print(f"  [LỖI] {name}: {e}")
            return False
    
    def combine_features(self, colors: np.ndarray, lbp_features: np.ndarray, 
                        hog_features: np.ndarray) -> np.ndarray:
        """
        Kết hợp 3 loại features thành 1 vector duy nhất
        
        Sử dụng trọng số từ config: Color(0.2) + LBP(0.3) + HOG(0.5)
        
        Args:
            colors: Color features
            lbp_features: LBP features
            hog_features: HOG features
            
        Returns:
            Combined features
        """
        # Chuẩn hóa L2 cho từng loại features
        colors_norm = self._l2_normalize(colors)
        lbp_norm = self._l2_normalize(lbp_features)
        hog_norm = self._l2_normalize(hog_features)
        
        # Kết hợp với trọng số
        combined = np.hstack([
            colors_norm * FEATURE_WEIGHTS["color"],
            lbp_norm * FEATURE_WEIGHTS["lbp"],
            hog_norm * FEATURE_WEIGHTS["hog"]
        ])
        
        logger.info(f"Combined features shape: {combined.shape}")
        return combined
    
    @staticmethod
    def _l2_normalize(features: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa L2 cho features
        
        Args:
            features: Features cần chuẩn hóa (N, D)
            
        Returns:
            Features đã chuẩn hóa
        """
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8  # Tránh chia cho 0
        return features / norms
    
    def load_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load features đã lưu từ file
        
        Returns:
            (colors, lbp_features, hog_features, labels)
        """
        logger.info("Đang load features từ file...")
        
        colors = np.load(str(COLOR_FEATURES_FILE))
        lbp_features = np.load(str(LBP_FEATURES_FILE))
        hog_features = np.load(str(HOG_FEATURES_FILE))
        labels = np.load(str(LABELS_FILE))
        
        logger.info(f"Đã load: Colors{colors.shape}, LBP{lbp_features.shape}, HOG{hog_features.shape}")
        
        return colors, lbp_features, hog_features, labels


if __name__ == "__main__":
    # Test FeatureManager
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    manager = FeatureManager()
    
    # Trích xuất features
    colors, lbp, hog, labels = manager.extract_all_features()
    
    print(f"\nKết quả:")
    print(f"  Colors: {colors.shape}")
    print(f"  LBP: {lbp.shape}")
    print(f"  HOG: {hog.shape}")
    print(f"  Labels: {labels.shape}")
