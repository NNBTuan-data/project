"""
SCRIPT 3: HUẤN LUYỆN MODEL SVM
"""
import sys
sys.dont_write_bytecode = True
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.svm_trainer import SVMTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Huấn luyện SVM model"""
    print("\n" + "="*70)
    print("BƯỚC 3: HUẤN LUYỆN MODEL SVM")
    print("="*70 + "\n")
    
    trainer = SVMTrainer()
    accuracy = trainer.train()
    
    print(f"\n✅ Model đã được huấn luyện!")
    print(f"   Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng")
    except Exception as e:
        logger.error(f"Lỗi: {e}", exc_info=True)
        print(f"\n❌ LỖI: {e}")
        sys.exit(1)
