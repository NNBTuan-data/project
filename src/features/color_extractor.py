import cv2
import logging
from ..config import COLOR_BINS
logger = logging.getLogger(__name__)
class ColorExtractor:
    def __init__(self, bins=None):
        self.bins = bins or COLOR_BINS
        logger.info(f"Khởi tạo ColorExtractor: bins={self.bins}")
    def extract(self, img_path: str) -> list:
        img = cv2.imread(str(img_path))
        if img is None:
            return [0.0] * (3 * self.bins)
        height, width, _ = img.shape
        hist_r = [0] * self.bins
        hist_g = [0] * self.bins
        hist_b = [0] * self.bins
        bin_size = 256 / self.bins
        for i in range(height):
            for j in range(width):
                b = img[i, j, 0]
                g = img[i, j, 1]
                r = img[i, j, 2]
                idx_r = int(r / bin_size)
                idx_g = int(g / bin_size)
                idx_b = int(b / bin_size)
                if idx_r >= self.bins: idx_r = self.bins - 1
                if idx_g >= self.bins: idx_g = self.bins - 1
                if idx_b >= self.bins: idx_b = self.bins - 1
                hist_r[idx_r] += 1
                hist_g[idx_g] += 1
                hist_b[idx_b] += 1
        hist = hist_r + hist_g + hist_b
        total = sum(hist)
        if total > 0:
            hist = [x / total for x in hist]    
        return hist
    def get_feature_dim(self) -> int:
        return 3 * self.bins
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    extractor = ColorExtractor()
    print(f"Color feature dimension: {extractor.get_feature_dim()}")
