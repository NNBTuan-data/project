
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib
import os

# ĐƯỜNG DẪN TUYỆT ĐỐI
BASE_DIR = Path(__file__).parent.parent
COLOR_FILE = BASE_DIR / "features" / "colors.npy"
LBP_FILE = BASE_DIR / "features" / "lbp.npy"
HOG_FILE = BASE_DIR / "features" / "hog.npy"
LABELS_FILE = BASE_DIR / "features" / "labels.npy"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

print("HUẤN LUYỆN SIÊU NHANH (< 1 GIÂY) – CHUẨN 100%")

# TẢI DỮ LIỆU
colors = np.load(COLOR_FILE)
lbp = np.load(LBP_FILE)
hog = np.load(HOG_FILE)
labels = np.load(LABELS_FILE)

print(f"   → TẢI: {len(labels)} ảnh (MÀU {colors.shape[1]}, LBP {lbp.shape[1]}, HOG {hog.shape[1]})")


# LOẠI LỚP 1 ẢNH
label_counts = Counter(labels)
valid_idx = [i for i, l in enumerate(labels) if label_counts[l] >= 2]
if len(valid_idx) < 10:
    print("LỖI: Không đủ dữ liệu!")
    exit()

colors, lbp, hog, labels = [arr[valid_idx] for arr in [colors, lbp, hog, labels]]
print(f"   → Sau lọc: {len(labels)} ảnh, {len(set(labels))} lớp")

# GỘP ĐẶC TRƯNG (COLOR 0.2, LBP 0.3, HOG 0.5)
X = np.hstack([
    colors * 0.2,
    lbp * 0.3,
    hog * 0.5
])
print(f"   → Vector cuối: {X.shape[1]} chiều (tối ưu)")

# CHIA DỮ LIỆU
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# HUẤN LUYỆN SIÊU NHANH
pipeline = Pipeline([
    ('scaler', StandardScaler()),           # Chuẩn hóa dữ liệu
    ('svm', SVC(kernel='linear', C=10.0))   # Linear + C cao
])

print("[HUẤN LUYỆN] SVM Linear + Chuẩn hóa...")
pipeline.fit(X_train, y_train)

# Đánh giá
acc = pipeline.score(X_test, y_test)
print(f"[HOÀN TẤT] Độ chính xác: {acc*100:.2f}%")

# LƯU MÔ HÌNH
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"[LƯU MÔ HÌNH] {MODEL_PATH}")