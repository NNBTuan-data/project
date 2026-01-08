import numpy as np
import logging

from ..config import HOG_PARAMS

logger = logging.getLogger(__name__)


class HOGExtractor:
    def __init__(self, orientations=None, pixels_per_cell=None, 
                 cells_per_block=None, block_norm=None):
        self.orientations = orientations or HOG_PARAMS["orientations"]
        self.pixels_per_cell = tuple(pixels_per_cell or HOG_PARAMS["pixels_per_cell"])
        self.cells_per_block = tuple(cells_per_block or HOG_PARAMS["cells_per_block"])
        self.block_norm = block_norm or HOG_PARAMS["block_norm"]
        
        logger.info(f"CUSTOM HOG: orientations={self.orientations}, "
                   f"cell={self.pixels_per_cell}, block={self.cells_per_block}")
    
    def compute_gradients(self, img: np.ndarray) -> tuple:
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        
        sobel_y = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)
        
        # Convolve thủ công (Manual 2D Convolution)
        h, w = img.shape
        # Pad ảnh để giữ nguyên kích thước (padding=1 do kernel 3x3)
        padded = np.pad(img, ((1, 1), (1, 1)), mode='symmetric').astype(np.float32)
        
        Gx = np.zeros_like(img, dtype=np.float32)
        Gy = np.zeros_like(img, dtype=np.float32)

        # Duyệt qua từng pixel (bỏ qua viền padding)
        for i in range(h):
            for j in range(w):
                # Lấy vùng 3x3 xung quanh pixel
                region = padded[i:i+3, j:j+3]
                
                # Nhân chập (Element-wise multiplication & sum)
                gx_val = np.sum(region * sobel_x)
                gy_val = np.sum(region * sobel_y)
                
                Gx[i, j] = gx_val
                Gy[i, j] = gy_val
        
        # Magnitude: sqrt(Gx^2 + Gy^2)
        magnitude = np.sqrt(Gx**2 + Gy**2)
        
        # Orientation: arctan(Gy / Gx) in degrees [0, 180)
        orientation = np.arctan2(Gy, Gx) * (180 / np.pi)
        orientation[orientation < 0] += 180  # Convert to [0, 180)
        
        logger.debug(f"Gradients computed: mag range [{magnitude.min():.2f}, {magnitude.max():.2f}]")
        
        return magnitude, orientation
    
    def compute_cell_histogram(self, magnitude: np.ndarray, 
                               orientation: np.ndarray) -> np.ndarray:
        hist = np.zeros(self.orientations, dtype=np.float32)
        bin_size = 180.0 / self.orientations  # 20 degrees per bin
        
        # Flatten arrays
        mag_flat = magnitude.flatten()
        ori_flat = orientation.flatten()
        
        # Weighted voting với linear interpolation
        for mag, angle in zip(mag_flat, ori_flat):
            if mag == 0:
                continue
            
            # Tìm 2 bins gần nhất
            bin_idx = angle / bin_size
            bin_low = int(np.floor(bin_idx)) % self.orientations
            bin_high = int(np.ceil(bin_idx)) % self.orientations
            
            # Linear interpolation weights
            weight_high = bin_idx - np.floor(bin_idx)
            weight_low = 1.0 - weight_high
            
            # Phân phối magnitude vào 2 bins
            hist[bin_low] += mag * weight_low
            hist[bin_high] += mag * weight_high
        
        return hist
    
    def normalize_block(self, block_hist: np.ndarray, eps=1e-5) -> np.ndarray:
        # Step 1: L2 normalization
        norm = np.sqrt(np.sum(block_hist**2) + eps**2)
        normalized = block_hist / norm
        
        if self.block_norm == 'L2-Hys':
            # Step 2: Clip values > 0.2
            normalized = np.clip(normalized, 0, 0.2)
            
            # Step 3: Re-normalize
            norm2 = np.sqrt(np.sum(normalized**2) + eps**2)
            normalized = normalized / norm2
        
        return normalized
    
    def extract(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        cell_h, cell_w = self.pixels_per_cell
        block_h, block_w = self.cells_per_block
        
        # Step 1: Compute gradients
        magnitude, orientation = self.compute_gradients(img)
        
        # Step 2: Calculate number of cells
        cells_y = h // cell_h
        cells_x = w // cell_w
        
        logger.debug(f"Image {h}x{w} → Grid {cells_y}x{cells_x} cells")
        
        # Step 3: Compute histogram cho mỗi cell
        cell_histograms = np.zeros((cells_y, cells_x, self.orientations), dtype=np.float32)
        
        for i in range(cells_y):
            for j in range(cells_x):
                # Extract cell
                y_start, y_end = i * cell_h, (i + 1) * cell_h
                x_start, x_end = j * cell_w, (j + 1) * cell_w
                
                mag_cell = magnitude[y_start:y_end, x_start:x_end]
                ori_cell = orientation[y_start:y_end, x_start:x_end]
                
                # Compute histogram
                cell_histograms[i, j] = self.compute_cell_histogram(mag_cell, ori_cell)
        
        # Step 4-5: Group into blocks và normalize
        blocks_y = cells_y - block_h + 1
        blocks_x = cells_x - block_w + 1
        
        features = []
        
        for i in range(blocks_y):
            for j in range(blocks_x):
                # Extract block (2x2 cells)
                block_cells = cell_histograms[i:i+block_h, j:j+block_w]
                
                # Concatenate cell histograms in block
                block_hist = block_cells.flatten()
                
                # Normalize block
                normalized = self.normalize_block(block_hist)
                
                features.append(normalized)
        
        # Step 6: Concatenate all blocks
        features = np.concatenate(features).astype(np.float32)
        
        logger.debug(f"HOG features shape: {features.shape}")
        
        return features


if __name__ == "__main__":
    # Test custom HOG implementation
    logging.basicConfig(level=logging.DEBUG)
    
    print("\n" + "="*70)
    print("TESTING CUSTOM HOG IMPLEMENTATION")
    print("="*70 + "\n")
    
    # Create test image
    test_img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    
    # Extract HOG features
    extractor = HOGExtractor()
    features = extractor.extract(test_img)
    
    print(f"   HOG features extracted successfully!")
    print(f"   Shape: {features.shape}")
    print(f"   Range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"   Mean: {features.mean():.4f}")
    
    # Expected dimension calculation
    h, w = 128, 128
    cell_size = 8
    cells_y, cells_x = h // cell_size, w // cell_size  # 16x16 cells
    blocks_y = cells_y - 2 + 1  # 15
    blocks_x = cells_x - 2 + 1  # 15
    expected_dim = blocks_y * blocks_x * (2 * 2 * 9)  # 15*15*36 = 8100
    
    print(f"\n   Expected dimension: {expected_dim}")
    print(f"   Actual dimension: {features.shape[0]}")
    
    if features.shape[0] == expected_dim:
        print("\n PASSED: Dimension matches expected!")
    else:
        print("\n FAILED: Dimension mismatch!")