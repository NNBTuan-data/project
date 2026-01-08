"""
SCRIPT 5: TRỰC QUAN HÓA ĐẶC TRƯNG

Lưu ý: Script này là wrapper đơn giản.
Để tạo visualization, sử dụng trực tiếp file cũ.
"""
import sys
sys.dont_write_bytecode = True
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)


def main():
    """Thông báo về visualization"""
    print("\n" + "="*70)
    print("BƯỚC 5: TRỰC QUAN HÓA ĐẶC TRƯNG")
    print("="*70 + "\n")
    
    print("⚠️  Lưu ý: Visualization script đang sử dụng file cũ.")
    print()
    print("📝 Để tạo visualization:")
    print()
    print("   cd d:/DA_PHUONG_TIEN/project/src_backup")
    print("   python visualize_features.py --dataset ../dataset --save_dir ../viz_output")
    print()
    print("Kết quả sẽ được lưu vào thư mục: viz_output/")
    print()
    print("Hoặc xem features trực tiếp:")
    print("   cd d:/DA_PHUONG_TIEN/project/src_backup")
    print("   python view_features.py")
    print()
    print("="*70)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng")
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        sys.exit(1)
