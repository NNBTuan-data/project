import sys
sys.dont_write_bytecode = True
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
def main():
    """Thông báo về evaluation"""
    print("\n" + "="*70)
    print("BƯỚC 4: ĐÁNH GIÁ MODEL")
    print("="*70 + "\n")
    print("⚠️  Lưu ý: Evaluation script đang sử dụng file cũ.")
    print()
    print("📝 Để đánh giá model, bạn có 2 cách:")
    print()
    print("Cách 1: Sử dụng file cũ trực tiếp")
    print("   cd d:/DA_PHUONG_TIEN/project/src_backup")
    print("   python evaluate.py --features ../features/features.npy --labels ../features/labels.npy --dataset ../dataset")
    print()
    print("Cách 2: Test trong GUI")
    print("   python scripts/6_run_gui.py")
    print("   → Chọn ảnh test và xem độ chính xác")
    print()
    print("="*70)
    print()
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Đã hủy bởi người dùng")
    except Exception as e:
        print(f"\n LỖI: {e}")
        sys.exit(1)
