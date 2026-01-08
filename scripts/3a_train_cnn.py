import sys
sys.dont_write_bytecode = True
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.cnn_trainer import CNNTrainer
from src.config import CNN_MODEL_TYPE, CNN_EPOCHS
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)
def main():
    print("\n" + "="*70)
    print(f"HUẤN LUYỆN MODEL CNN - {CNN_MODEL_TYPE.upper()}")
    print("="*70 + "\n")
    print()
    trainer = CNNTrainer(model_type=CNN_MODEL_TYPE)
    trainer.build_model()
    trainer.compile_model()
    history = trainer.train(
        epochs=CNN_EPOCHS,
        fine_tune=True,
        fine_tune_epochs=10
    )
    best_acc = max(history.history['val_accuracy'])
    
    print("\n" + "="*70)
    print(f"   HOÀN TẤT TRAINING CNN!")
    print(f"   Best Validation Accuracy: {best_acc*100:.2f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Training đã bị hủy bởi người dùng")
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        print(f"\n LỖI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
