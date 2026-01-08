import sys
sys.dont_write_bytecode = True
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.gui.main_window import run_gui
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)
def main():
    print("\n" + "="*70)
    print("BƯỚC 6: CHẠY GUI APPLICATION")
    print("="*70 + "\n")
    print("Đang khởi động giao diện...")
    logger.info("Starting GUI application")
    run_gui()
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Đã đóng ứng dụng")
        logger.info("Application closed by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\n LỖI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
