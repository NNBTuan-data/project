import os
from pathlib import Path

# ĐƯỜNG DẪN CƠ BẢN
BASE_DIR = Path(__file__).parent.parent.resolve()
DATASET_DIR = BASE_DIR / "dataset"
FEATURES_DIR = BASE_DIR / "features"
MODEL_DIR = BASE_DIR / "model"
VIZ_OUTPUT_DIR = BASE_DIR / "viz_output"

# Đảm bảo các thư mục tồn tại
for directory in [DATASET_DIR, FEATURES_DIR, MODEL_DIR, VIZ_OUTPUT_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# FILE PATHS
COLOR_FEATURES_FILE = FEATURES_DIR / "colors.npy"
LBP_FEATURES_FILE = FEATURES_DIR / "lbp.npy"
HOG_FEATURES_FILE = FEATURES_DIR / "hog.npy"
LABELS_FILE = FEATURES_DIR / "labels.npy"
COMBINED_FEATURES_FILE = FEATURES_DIR / "features.npy"

# Model files
SVM_MODEL_PATH = MODEL_DIR / "model.pkl"

# DATASET CONFIGURATION
CLASSES = ["chair", "table", "bottle", "cup", "man", "woman", "child", "people"]
NUM_IMAGES_PER_CLASS = 100
IMAGE_SIZE = (128, 128)

# DATASET DOWNLOAD PARAMS
MAX_DOWNLOAD_WORKERS = 5
MIN_IMAGE_SIZE = 5000
DOWNLOAD_TIMEOUT = 15

# FEATURE EXTRACTION PARAMS
# Color Histogram
COLOR_BINS = 32  # Số bins cho mỗi kênh màu (RGB)

# LBP (Local Binary Pattern)
LBP_PARAMS = {
    "P": 8,           # Số điểm láng giềng
    "R": 1,           # Bán kính
    "grid_x": 8,      # Rollback về 8×8 (cân bằng)
    "grid_y": 8,      # Đơn giản hơn, ít overfitting hơn
    "bins": 256       # Số bins trong histogram
}

# HOG (Histogram of Oriented Gradients)
HOG_PARAMS = {
    "orientations": 9,           # Rollback về 9 (đủ dùng)
    "pixels_per_cell": (8, 8),   # Kích thước cell
    "cells_per_block": (2, 2),   # Số cells trong block
    "block_norm": "L2-Hys"       # Phương pháp chuẩn hóa
}

# MODEL TRAINING PARAMS
# Trọng số kết hợp đặc trưng (tổng = 1.0)
# CÂN BẰNG: Phân bố đều hơn
FEATURE_WEIGHTS = {
    "color": 0.2,   # 20% - Màu sắc
    "lbp": 0.35,    # 35% - Kết cấu (tăng - quan trọng!)
    "hog": 0.45     # 45% - Hình dạng
}

# SVM Hyperparameters
# CÂN BẰNG: Linear kernel + C vừa phải
SVM_PARAMS = {
    "kernel": "linear",  # Linear đơn giản, hiệu quả
    "C": 5.0             # Vừa phải: không quá lớn (overfitting) không quá nhỏ (underfitting)
}

# Train/Test split
TEST_SIZE = 0.2        # Rollback về 20%
RANDOM_STATE = 42      # Seed cho reproducibility

# RECOGNITION PARAMS
# Ngưỡng similarity tối thiểu
SIMILARITY_THRESHOLD = 0.75

# Số kết quả top trả về
TOP_K_RESULTS = 5

# CNN / DEEP LEARNING PARAMS (NEW)
# Model type
CNN_MODEL_TYPE = "mobilenet"  # Đổi từ "resnet50" → "mobilenet" (nhẹ hơn, phù hợp data nhỏ)
CNN_MODEL_PATH = MODEL_DIR / "cnn_model.h5"
CNN_WEIGHTS_PATH = MODEL_DIR / "cnn_weights.h5"

# Training hyperparameters
CNN_EPOCHS = 80              # Tăng từ 50 → 80 epochs (quick win)
CNN_BATCH_SIZE = 32          # Batch size
CNN_LEARNING_RATE = 0.0001   # Learning rate (nhỏ cho fine-tuning)
CNN_VALIDATION_SPLIT = 0.2   # 20% validation

# Fine-tuning
CNN_UNFREEZE_LAYERS = 30     # Giảm từ 50 → 30 (MobileNet nhẹ hơn)

# Data Augmentation (tăng mạnh để tạo thêm data)
CNN_AUGMENTATION = {
    "rotation_range": 30,        # Tăng từ 20 → 30 degrees
    "width_shift_range": 0.3,    # Tăng từ 0.2 → 0.3
    "height_shift_range": 0.3,   # Tăng từ 0.2 → 0.3
    "shear_range": 0.2,          # Tăng từ 0.15 → 0.2
    "zoom_range": 0.25,          # Tăng từ 0.15 → 0.25
    "horizontal_flip": True,     # Flip ngang
    "brightness_range": [0.8, 1.2],  # Thêm brightness variation
    "fill_mode": "nearest"       # Fill mode cho pixels mới
}

# Early stopping & callbacks
CNN_EARLY_STOPPING_PATIENCE = 12   # Tăng từ 7 → 12 (cho phép train lâu hơn)
CNN_REDUCE_LR_PATIENCE = 3         # Giảm LR nếu không cải thiện sau 3 epochs
CNN_REDUCE_LR_FACTOR = 0.5         # Giảm LR xuống 50%

# ================================
# VISUALIZATION PARAMS
# ================================
# DPI cho ảnh export
VIZ_DPI = 300

# Kích thước figure
VIZ_FIGURE_SIZE = (32, 20)

# ================================
# GUI CONFIGURATION
# ================================
# Màu sắc - Modern Gradient Theme
UI_COLORS = {
    "bg": "#0f172a",          # Dark blue background
    "card": "#1e293b",        # Slate card
    "primary": "#3b82f6",     # Bright blue
    "secondary": "#8b5cf6",   # Purple
    "success": "#10b981",     # Green
    "danger": "#ef4444",      # Red
    "text_dark": "#f1f5f9",   # Light text
    "text_light": "#94a3b8",  # Muted text
    "border": "#334155",      # Border
    "accent": "#06b6d4"       # Cyan accent
}

# Kích thước cửa sổ - TO HƠN
GUI_WINDOW_SIZE = "1600x900"

# Kích thước thumbnail
THUMBNAIL_SIZE = (120, 120)

# Kích thước preview - TO HƠN
PREVIEW_SIZE = (500, 500)


# ================================
# LOGGING CONFIGURATION
# ================================
# Cấu hình logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(BASE_DIR / "app.log"),
            "mode": "a"
        }
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}


# UTILITY FUNCTIONS
def get_class_index(class_name: str) -> int:
    """Lấy index của class từ tên"""
    try:
        return CLASSES.index(class_name)
    except ValueError:
        raise ValueError(f"Class '{class_name}' không tồn tại trong CLASSES")

def get_class_name(class_index: int) -> str:
    """Lấy tên class từ index"""
    if 0 <= class_index < len(CLASSES):
        return CLASSES[class_index]
    raise ValueError(f"Class index {class_index} nằm ngoài phạm vi [0, {len(CLASSES)-1}]")

def print_config():
    """In ra cấu hình hiện tại"""
    print("=" * 70)
    print("CẤU HÌNH HỆ THỐNG NHẬN DIỆN ẢNH")
    print("=" * 70)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Classes ({len(CLASSES)}): {', '.join(CLASSES)}")
    print(f"Feature Weights: Color={FEATURE_WEIGHTS['color']}, "
          f"LBP={FEATURE_WEIGHTS['lbp']}, HOG={FEATURE_WEIGHTS['hog']}")
    print(f"SVM Kernel: {SVM_PARAMS['kernel']}, C={SVM_PARAMS['C']}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()
