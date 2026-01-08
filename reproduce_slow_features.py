
import time
import numpy as np
import logging
import cv2
from src.features.color_extractor import ColorExtractor
from src.features.lbp_extractor import LBPExtractor
from src.features.hog_extractor import HOGExtractor

def test_performance():
    # Create a dummy image (RGB)
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    path = "dummy_img.jpg"
    cv2.imwrite(path, img)
    
    print("Starting performance test on 128x128 image...")
    
    # Test Color
    start = time.time()
    color_ext = ColorExtractor()
    color_ext.extract(path)
    print(f"Color Extraction: {time.time() - start:.4f}s")
    
    # Test LBP
    start = time.time()
    lbp_ext = LBPExtractor()
    lbp_ext.extract(img_gray)
    print(f"LBP Extraction: {time.time() - start:.4f}s")
    
    # Test HOG
    start = time.time()
    hog_ext = HOGExtractor()
    hog_ext.extract(img_gray)
    print(f"HOG Extraction: {time.time() - start:.4f}s")

    # Test larger image
    img_large = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img_large_gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
    path_large = "dummy_img_large.jpg"
    cv2.imwrite(path_large, img_large)

    print("\nStarting performance test on 512x512 image...")
    
    # Test Color
    start = time.time()
    color_ext.extract(path_large)
    print(f"Color Extraction: {time.time() - start:.4f}s")
    
    # Test LBP
    start = time.time()
    lbp_ext.extract(img_large_gray)
    print(f"LBP Extraction: {time.time() - start:.4f}s")
    
    # Test HOG
    start = time.time()
    hog_ext.extract(img_large_gray)
    print(f"HOG Extraction: {time.time() - start:.4f}s")

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    test_performance()
