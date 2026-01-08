import sys
sys.dont_write_bytecode = True
import logging
import logging.config
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset_downloader import DatasetDownloader
from src.config import CLASSES, NUM_IMAGES_PER_CLASS
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)
def main():
    print("Bước 1: Tải ảnh theo class từ internet")
    downloader = DatasetDownloader()
    results = downloader.download_all(
        classes=CLASSES,
        num_images=NUM_IMAGES_PER_CLASS
    )
    total = sum(results.values())
    print(f"Thống kê ảnh đã tải:")
    for cls, count in results.items():
        print(f"   {cls:15s}: {count:3d} ảnh")
    print(f"   {'Tổng':15s}: {total:3d} ảnh")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Đã hủy bởi người dùng")
    except Exception as e:
        logger.error(f"Lỗi: {e}", exc_info=True)
        print(f"\n Lỗi: {e}")
        sys.exit(1)
