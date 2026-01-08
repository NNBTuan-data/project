# src/view_features.py
import numpy as np
import pandas as pd
import os

# ĐƯỜNG DẪN TUYỆT ĐỐI – ỔN ÁP
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "features", "features.npy")
LABELS_PATH = os.path.join(BASE_DIR, "features", "labels.npy")

print("=" * 70)
print("       XEM CHI TIẾT IMAGE INDEX (CSDL ĐA PHƯƠNG TIỆN)")
print("=" * 70)

# TẢI DỮ LIỆU
features = np.load(FEATURES_PATH)
labels = np.load(LABELS_PATH)

print(f"Kích thước ma trận đặc trưng (LBP + HOG): {features.shape}")
print(f"Số lượng ảnh trong CSDL: {len(labels)}")
print(f"Số lớp (nhãn): {len(np.unique(labels))}")
print(f"Các lớp: {sorted(np.unique(labels))}")

# HIỂN THỊ 100 MẪU ĐẦU TIÊN – CHỈ 5 ĐẶC TRƯNG ĐẦU
df = pd.DataFrame(features[:, :5], columns=[f"feat_{i}" for i in range(5)])
df.insert(0, "label", labels)

print("\n100 MẪU ĐẦU TIÊN (5 đặc trưng đầu tiên):")
print("-" * 60)
print(df.head(100).to_string(index=False))

print("\nHOÀN TẤT XEM IMAGE INDEX!")
print("=" * 70)