"""
SCRIPT 3B: TRAIN SVM MODEL (Classical ML)
Đã rename từ 3_train_model.py để phân biệt với CNN
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
    """Huấn luyện SVM model (Classical ML)"""
    print("\n" + "="*70)
    print("HUẤN LUYỆN MODEL SVM (CLASSICAL ML)")
    print("="*70 + "\n")
    
    print("⚠️  Lưu ý: Script này train Classical ML (LBP+HOG+SVM)")
    print("   Để train CNN (accuracy cao hơn), dùng: python scripts/3a_train_cnn.py")
    print()
    
    trainer = SVMTrainer()
    accuracy = trainer.train()
    
    print(f"\n✅ Model SVM đã được huấn luyện!")
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
