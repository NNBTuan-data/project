# src/extract_features.py
import os
import argparse
import numpy as np
from utils import list_images, read_image, extract_color_histogram, extract_lbp_histogram, extract_hog_feature
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def safe_save(data, path):
    try:
        arr = np.array(data)
        np.save(path, arr)
        print(f"[OK] Lưu thành công: {os.path.basename(path)} ({arr.nbytes / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"[LỖI] Lưu thất bại {os.path.basename(path)}: {e}")
        return False

def main(args):
    dataset_path = os.path.join(BASE_DIR, args.dataset)
    if not os.path.exists(dataset_path):
        print(f"[LỖI] Không tìm thấy dataset: {dataset_path}")
        return

    paths, labels = list_images(dataset_path)
    print(f"TRÍCH XUẤT ĐẶC TRƯNG TỪ {len(paths)} ẢNH...")

    color_list, lbp_list, hog_list = [], [], []

    for path in tqdm(paths, desc="Trích xuất", unit="ảnh"):
        img = read_image(path, size=(args.size, args.size))
        color_list.append(extract_color_histogram(path))
        lbp_list.append(extract_lbp_histogram(img))
        hog_list.append(extract_hog_feature(img))

    out_dir = os.path.join(BASE_DIR, "features")

    # LƯU TỪNG FILE RIÊNG
    safe_save(color_list, os.path.join(out_dir, "colors.npy"))
    safe_save(lbp_list,   os.path.join(out_dir, "lbp.npy"))
    safe_save(hog_list,   os.path.join(out_dir, "hog.npy"))
    safe_save(labels,     os.path.join(out_dir, "labels.npy"))

    print(f"\nHOÀN TẤT! Kiểm tra tại: features/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    main(args)