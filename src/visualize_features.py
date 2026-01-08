# src/visualize_features.py
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import list_images, read_image
from pathlib import Path
from tqdm import tqdm
import cv2
from skimage.feature import local_binary_pattern
from multiprocessing import Pool

plt.ioff()
plt.switch_backend('Agg')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'figure.facecolor': '#ffffff',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.4,
})

def get_class_names(dataset_dir):
    classes = [d.name for d in Path(dataset_dir).iterdir() if d.is_dir()]
    classes.sort()
    return classes

def lbp_3x3(img):
    # Sử dụng skimage để tăng tốc
    return local_binary_pattern(img, P=8, R=1, method='uniform').astype(np.uint8)

def color_based_segmentation(gray):
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    result = img_rgb.copy()
    result[mask > 0] = [200, 200, 200]
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

def create_visualization(args):
    img_path, gray, color_hist, lbp_features, hog_features, labels, class_names, idx, total, save_dir = args
    plt.close('all')
    fig = plt.figure(figsize=(32, 20))  # Tăng kích thước ảnh
    gs = GridSpec(3, 4, figure=fig, wspace=0.5, hspace=0.6)  # Tăng khoảng cách

    # 1. ẢNH GỐC
    ax0 = fig.add_subplot(gs[:, 0])
    try:
        orig = Image.open(img_path).convert("RGB")
        w, h = orig.size
    except:
        w, h = 600, 600
        orig = Image.new("RGB", (w, h), (240, 240, 240))
        from PIL import ImageDraw
        d = ImageDraw.Draw(orig)
        d.text((w//2 - 100, h//2), "KHÔNG TẢI ĐƯỢC", fill=(200, 0, 0))
    ax0.imshow(orig)
    ax0.set_title(f"1. ẢNH GỐC\n{Path(img_path).name}\n{w:,}×{h:,} px", fontsize=18, weight='bold', pad=20)
    ax0.axis('off')

    # 2. TIỀN XỬ LÝ
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(gray, cmap='gray')
    ax1.set_title("2. TIỀN XỬ LÝ\nGrayscale + Resize 128×128", fontsize=16, weight='bold')
    ax1.axis('off')

    # 3. LBP 3×3
    ax2 = fig.add_subplot(gs[0, 2])
    lbp_map = lbp_3x3(gray)
    ax2.imshow(lbp_map, cmap='gray')
    ax2.set_title("3. LBP 3×3\nHoa văn – Mã nhị phân", fontsize=16, weight='bold')
    ax2.axis('off')

    # 4. BIÊN
    ax3 = fig.add_subplot(gs[0, 3])
    edges = cv2.Canny(gray, 50, 150)
    ax3.imshow(edges, cmap='gray')
    ax3.set_title("4. PHÁT HIỆN BIÊN\nCanny Edge – Hình dạng", fontsize=16, weight='bold')
    ax3.axis('off')

    # 5. CUMULATIVE HISTOGRAM MÀU
    ax4 = fig.add_subplot(gs[1, 1])
    cum_hist = np.cumsum(color_hist)
    ax4.stairs(cum_hist, np.arange(len(cum_hist)+1), fill=True, color='#e74c3c', alpha=0.8)
    ax4.set_title("5. HISTOGRAM TÍCH LŨY MÀU", fontsize=16, weight='bold')
    ax4.set_xlabel("Bin màu")
    ax4.set_ylabel("Tích lũy")
    ax4.grid(True, alpha=0.5)

    # 6. HISTOGRAM LBP
    ax5 = fig.add_subplot(gs[1, 2])
    lbp_hist = np.histogram(lbp_map, bins=256, range=(0, 256))[0]
    ax5.stairs(lbp_hist, np.arange(257), fill=True, color='#3498db', alpha=0.8)
    ax5.set_title("6. HISTOGRAM LBP\nKết cấu – Hoa văn", fontsize=16, weight='bold')
    ax5.set_xlabel("Mã LBP")
    ax5.set_ylabel("Tần suất")
    ax5.grid(True, alpha=0.5)

    # 7. HOG
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.hist(hog_features, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax6.set_title(f"7. HOG\nĐộ dài: {len(hog_features):,}", fontsize=16, weight='bold')
    ax6.set_xlabel("Giá trị")
    ax6.set_ylabel("Tần suất")
    ax6.grid(True, alpha=0.5)

    # 8. TÁCH NỀN THEO MÀU
    ax7 = fig.add_subplot(gs[2, 1:3])
    segmented = color_based_segmentation(gray)
    ax7.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    ax7.set_title("8. TÁCH NỀN & ĐỐI TƯỢNG\nNền trắng → Xám", fontsize=16, weight='bold')
    ax7.axis('off')

    class_idx = labels[idx]
    class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
    fig.suptitle(f"3 HƯỚNG TÌM KIẾM ẢNH • Ảnh {idx+1:04d}/{total:04d} • LỚP: {class_name.upper()}", 
                 fontsize=22, weight='bold', y=0.96)
    fig.text(0.5, 0.02,
             "MÀU (Histogram + Tách nền) | HÌNH DẠNG (Edge + HOG) | HOA VĂN (LBP + Histogram)",
             ha='center', fontsize=15, style='italic', color='#2c3e50')

    os.makedirs(save_dir, exist_ok=True)
    filename = f"VISUAL_{idx+1:04d}_{class_name.upper()}_{Path(img_path).stem}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, facecolor='white')
    plt.close(fig)
    return f"  [LƯU] {filename}"

def main(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    colors_file = os.path.join(BASE_DIR, "features", "colors.npy")
    lbp_file    = os.path.join(BASE_DIR, "features", "lbp.npy")
    hog_file    = os.path.join(BASE_DIR, "features", "hog.npy")
    labels_file = os.path.join(BASE_DIR, "features", "labels.npy")
    dataset_dir = os.path.join(BASE_DIR, args.dataset)
    output_dir  = os.path.join(BASE_DIR, args.save_dir)

    class_names = get_class_names(dataset_dir)
    print(f"TÌM THẤY {len(class_names)} LỚP: {class_names}")

    try:
        colors = np.load(colors_file)
        lbp_features = np.load(lbp_file)
        hog_features = np.load(hog_file)
        labels = np.load(labels_file)
        print(f"TẢI: {len(colors)} màu, {len(lbp_features)} LBP, {len(hog_features)} HOG, {len(labels)} nhãn")
    except Exception as e:
        print(f"[LỖI] File .npy: {e}")
        return

    paths, _ = list_images(dataset_dir)
    N = min(len(paths), len(colors), len(lbp_features), len(hog_features), len(labels))
    paths, colors, lbp_features, hog_features, labels = paths[:N], colors[:N], lbp_features[:N], hog_features[:N], labels[:N]

    print(f"\nBẮT ĐẦU VISUALIZE {N} ẢNH – 3 HƯỚNG TÌM KIẾM")
    print(f"   Lưu tại: {args.save_dir}\n")

    # Song song hóa với multiprocessing
    with Pool() as pool:
        tasks = [(paths[i], read_image(paths[i], size=(128, 128)), colors[i], lbp_features[i], hog_features[i], labels, class_names, i, N, os.path.join(output_dir, class_names[labels[i]] if labels[i] < len(class_names) else f"class_{labels[i]}")) 
                 for i in range(N)]
        results = list(tqdm(pool.imap(create_visualization, tasks), total=N, desc="Visualize", unit="ảnh", colour="cyan"))

    for result in results:
        print(result)

    print(f"\nHOÀN TẤT! Xem tại: {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--save_dir", type=str, default="viz_output")
    args = parser.parse_args()
    main(args)