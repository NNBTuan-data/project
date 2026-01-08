import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
from ..config import IMAGE_SIZE
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = IMAGE_SIZE):
        self.target_size = target_size
        logger.info(f"Khởi tạo ImageProcessor (Tiền xử lý ảnh bao gồm: làm xám, resize, làm mờ, cân bằng histogram) với target_size={target_size}")
    
    def read_image(self, path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if size is None:
            size = self.target_size
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, size)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.equalizeHist(img)
        logger.debug(f"Đã xử lý ảnh: {Path(path).name}, shape={img.shape}")
        return img
    @staticmethod
    def normalize_histogram(hist: np.ndarray) -> np.ndarray:
        hist = np.array(hist, dtype=np.float32)
        s = hist.sum()
        if s > 0:
            hist = hist / s
        return hist
    def read_color_image(self, path: str) -> np.ndarray:
        img = cv2.imread(str(path))

        # Chuyển từ BGR (OpenCV format) sang RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logger.debug(f"Đã đọc ảnh màu: {Path(path).name}, shape={img.shape}")
        return img
    
    def batch_read_images(self, paths: list, show_progress: bool = True) -> list:
        images = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(paths, desc="Đọc ảnh", unit="ảnh")
            except ImportError:
                logger.warning("Không có tqdm, không hiển thị progress bar")
                iterator = paths
        else:
            iterator = paths
        for path in iterator:
            try:
                img = self.read_image(path)
                images.append(img)
            except Exception as e:
                logger.warning(f"Bỏ qua ảnh lỗi: {path} - {e}")
                continue
        
        logger.info(f"Đã đọc {len(images)}/{len(paths)} ảnh thành công")
        return images
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    processor = ImageProcessor()
    print(f"ImageProcessor initialized with size: {processor.target_size}")
