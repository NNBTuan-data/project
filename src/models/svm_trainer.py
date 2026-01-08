"""
Module huấn luyện SVM model
Chuyển đổi từ train.py
"""
import logging
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from ..config import (FEATURES_DIR, MODEL_DIR, COLOR_FEATURES_FILE,
                      LBP_FEATURES_FILE, HOG_FEATURES_FILE, LABELS_FILE,
                      SVM_MODEL_PATH, SVM_PARAMS, TEST_SIZE, RANDOM_STATE,
                      FEATURE_WEIGHTS)

logger = logging.getLogger(__name__)


class SVMTrainer:
    """
    Class huấn luyện SVM model cho nhận diện ảnh
    """
    
    def __init__(self, model_path: Path = None):
        """
        Khởi tạo SVMTrainer
        
        Args:
            model_path: Đường dẫn lưu model
        """
        self.model_path = model_path or SVM_MODEL_PATH
        self.pipeline = None
        
        # Tạo thư mục model nếu chưa tồn tại
        self.model_path.parent.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Khởi tạo SVMTrainer: model_path={self.model_path}")
    
    def load_features(self) -> tuple:
        """
        Load features từ file .npy
        
        Returns:
            (colors, lbp, hog, labels)
        """
        logger.info("Đang load features...")
        
        colors = np.load(str(COLOR_FEATURES_FILE))
        lbp = np.load(str(LBP_FEATURES_FILE))
        hog = np.load(str(HOG_FEATURES_FILE))
        labels = np.load(str(LABELS_FILE))
        
        logger.info(f"Đã load: Colors{colors.shape}, LBP{lbp.shape}, HOG{hog.shape}, Labels{labels.shape}")
        
        return colors, lbp, hog, labels
    
    def filter_small_classes(self, colors: np.ndarray, lbp: np.ndarray, 
                            hog: np.ndarray, labels: np.ndarray, 
                            min_samples: int = 2) -> tuple:
        """
        Lọc bỏ các class có ít hơn min_samples
        
        Lý do: train_test_split cần ít nhất 2 mẫu để chia
        
        Args:
            colors, lbp, hog, labels: Features và labels
            min_samples: Số mẫu tối thiểu
            
        Returns:
            (colors_filtered, lbp_filtered, hog_filtered, labels_filtered)
        """
        label_counts = Counter(labels)
        valid_idx = [i for i, l in enumerate(labels) if label_counts[l] >= min_samples]
        
        if len(valid_idx) < 10:
            logger.error("Không đủ dữ liệu sau khi lọc!")
            raise ValueError("Không đủ dữ liệu để huấn luyện!")
        
        colors_filtered = colors[valid_idx]
        lbp_filtered = lbp[valid_idx]
        hog_filtered = hog[valid_idx]
        labels_filtered = labels[valid_idx]
        
        logger.info(f"Sau lọc: {len(labels_filtered)} ảnh, {len(set(labels_filtered))} classes")
        print(f"   → Sau lọc: {len(labels_filtered)} ảnh, {len(set(labels_filtered))} lớp")
        
        return colors_filtered, lbp_filtered, hog_filtered, labels_filtered
    
    def combine_features(self, colors: np.ndarray, lbp: np.ndarray, hog: np.ndarray) -> np.ndarray:
        """
        Kết hợp features với trọng số tối ưu
        
        Args:
            colors, lbp, hog: Features
            
        Returns:
            Combined features
        """
        X = np.hstack([
            colors * FEATURE_WEIGHTS["color"],
            lbp * FEATURE_WEIGHTS["lbp"],
            hog * FEATURE_WEIGHTS["hog"]
        ])
        
        logger.info(f"Vector kết hợp: {X.shape[1]} chiều")
        print(f"   → Vector cuối: {X.shape[1]} chiều")
        
        return X
    
    def train(self, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> float:
        """
        Huấn luyện SVM model
        
        Pipeline:
        1. Load features
        2. Lọc classes nhỏ
        3. Kết hợp features
        4. Chia train/test
        5. Train SVM
        6. Đánh giá
        7. Lưu model
        
        Args:
            test_size: Tỉ lệ test set
            random_state: Random seed
            
        Returns:
            Accuracy trên test set
        """
        logger.info("BẮT ĐẦU HUẤN LUYỆN SVM")
        print(f"\n{'='*70}")
        print("HUẤN LUYỆN MODEL SVM")
        print(f"{'='*70}\n")
        
        # Bước 1: Load features
        colors, lbp, hog, labels = self.load_features()
        print(f"   → TẢI: {len(labels)} ảnh (MÀU {colors.shape[1]}, LBP {lbp.shape[1]}, HOG {hog.shape[1]})")
        
        # Bước 2: Lọc classes nhỏ
        colors, lbp, hog, labels = self.filter_small_classes(colors, lbp, hog, labels)
        
        # Bước 3: Kết hợp features
        X = self.combine_features(colors, lbp, hog)
        
        # Bước 4: Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels  # Đảm bảo tỉ lệ các class như nhau
        )
        
        logger.info(f"Train: {len(X_train)} ảnh, Test: {len(X_test)} ảnh")
        print(f"   → Train/Test: {len(X_train)}/{len(X_test)} ảnh\n")
        
        # Bước 5: Train SVM
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Chuẩn hóa dữ liệu (mean=0, std=1)
            ('svm', SVC(
                kernel=SVM_PARAMS['kernel'],
                C=SVM_PARAMS['C'],
                probability=True  # Cho phép predict_proba
            ))
        ])
        
        logger.info(f"[HUẤN LUYỆN] SVM {SVM_PARAMS['kernel']} kernel, C={SVM_PARAMS['C']}...")
        print(f"[HUẤN LUYỆN] SVM {SVM_PARAMS['kernel']} + Chuẩn hóa...")
        
        self.pipeline.fit(X_train, y_train)
        
        # Bước 6: Đánh giá
        train_acc = self.pipeline.score(X_train, y_train)
        test_acc = self.pipeline.score(X_test, y_test)
        
        logger.info(f"Accuracy: Train={train_acc*100:.2f}%, Test={test_acc*100:.2f}%")
        print(f"[HOÀN TẤT] Độ chính xác:")
        print(f"   → Train: {train_acc*100:.2f}%")
        print(f"   → Test: {test_acc*100:.2f}%\n")
        
        # Bước 7: Lưu model
        self.save_model()
        
        print(f"{'='*70}\n")
        
        return test_acc
    
    def save_model(self) -> None:
        """Lưu model đã train"""
        if self.pipeline is None:
            logger.warning("Chưa có model để lưu!")
            return
        
        joblib.dump(self.pipeline, str(self.model_path))
        logger.info(f"[LƯU MODEL] {self.model_path}")
        print(f"[LƯU MODEL] {self.model_path}")
    
    def load_model(self) -> Pipeline:
        """
        Load model đã lưu
        
        Returns:
            Pipeline đã train
        """
        if not self.model_path.exists():
            logger.error(f"Không tìm thấy model tại {self.model_path}")
            raise FileNotFoundError(f"Model không tồn tại: {self.model_path}")
        
        self.pipeline = joblib.load(str(self.model_path))
        logger.info(f"Đã load model từ {self.model_path}")
        
        return self.pipeline


if __name__ == "__main__":
    # Test SVMTrainer
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    trainer = SVMTrainer()
    accuracy = trainer.train()
    
    print(f"\n✅ Model đã được huấn luyện với accuracy: {accuracy*100:.2f}%")
