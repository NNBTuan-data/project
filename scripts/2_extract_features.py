import sys
sys.dont_write_bytecode = True
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.feature_manager import FeatureManager
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)
def main():
    print("Bước 2: Trích xuất đặc trưng")
    manager = FeatureManager()
    colors, lbp, hog, labels = manager.extract_all_features(show_progress=True)
    print(f"   Hoàn tất!")
    print(f"   Colors: {colors.shape}")
    print(f"   LBP: {lbp.shape}")
    print(f"   HOG: {hog.shape}")
    print(f"   Labels: {labels.shape}")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Đã hủy bởi người dùng")
    except Exception as e:
        logger.error(f"Lỗi: {e}", exc_info=True)
        print(f"\n  Lỗi: {e}")
        sys.exit(1)
