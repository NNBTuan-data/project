import logging
from typing import Tuple
from ..config import LBP_PARAMS

logger = logging.getLogger(__name__)


class LBPExtractor:
    def __init__(self, P: int = None, R: int = None, 
                 grid_x: int = None, grid_y: int = None, bins: int = None):
        self.P = P or LBP_PARAMS["P"]
        self.R = R or LBP_PARAMS["R"]
        self.grid_x = grid_x or LBP_PARAMS["grid_x"]
        self.grid_y = grid_y or LBP_PARAMS["grid_y"]
        self.bins = bins or LBP_PARAMS["bins"]
        
        logger.info(f"Khởi tạo LBPExtractor: P={self.P}, R={self.R}, "
                   f"grid=({self.grid_x}x{self.grid_y}), bins={self.bins}")
    
    def compute_lbp_image(self, img) -> list:
        h, w = img.shape
        lbp = [[0] * w for _ in range(h)]
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                    (1, 1), (1, 0), (1, -1), (0, -1)]

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center = img[y, x]  # Giá trị pixel trung tâm
                code = 0  # Mã nhị phân 8-bit
                
                # So sánh với 8 láng giềng
                for i, (dy, dx) in enumerate(neighbors):
                    neighbor = img[y + dy, x + dx]
                    
                    # Nếu láng giềng >= trung tâm → set bit thứ i = 1
                    if neighbor >= center:
                        code |= (1 << i)
                
                lbp[y][x] = code  # Lưu mã LBP (0-255)
        
        return lbp

    def compute_lbp_histogram(self, img) -> list:
        # Bước 1: Tính LBP cho toàn bộ ảnh
        lbp = self.compute_lbp_image(img)
        h, w = img.shape
        
        # Bước 2: Chia ảnh thành lưới (thủ công)
        # Kích thước mỗi ô
        cell_h = h // self.grid_y
        cell_w = w // self.grid_x
        
        features = []
        
        # Bước 3: Tính histogram cho từng ô
        for i in range(self.grid_y):
            for j in range(self.grid_x):
                # Xác định vùng tọa độ của ô
                y_start = i * cell_h
                y_end = (i + 1) * cell_h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w
                hist = [0] * self.bins
                
                count = 0
                for r in range(y_start, y_end):
                    for c in range(x_start, x_end):
                        if r < h and c < w:
                            val = lbp[r][c]
                            bin_idx = int(val * self.bins / 256)
                            if bin_idx >= self.bins: bin_idx = self.bins - 1
                            
                            hist[bin_idx] += 1
                            count += 1
                if count > 0:
                    hist = [x / count for x in hist]
                features.extend(hist)
        
        return features

    def extract(self, img) -> list:
        return self.compute_lbp_histogram(img)
        
    def get_feature_dim(self) -> int:
        return self.grid_x * self.grid_y * self.bins


if __name__ == "__main__":
    # Test LBPExtractor
    logging.basicConfig(level=logging.DEBUG)
    import numpy as np # Import numpy only within if __name__ == "__main__" for testing
    
    extractor = LBPExtractor()
    test_img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    features = extractor.extract(test_img)
    
    print(f"LBP feature dimension: {extractor.get_feature_dim()}")
    print(f"Actual features length: {len(features)}")
