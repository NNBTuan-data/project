import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
from recognize import recognize
import os
import joblib

def evaluate(features_file, labels_file, dataset_folder, test_folder=None, size=(128, 128), threshold=0.75):
    print("Đang tải dữ liệu đặc trưng và nhãn...")
    feats = np.load(features_file)
    labels = np.load(labels_file)

    # Nếu có test_folder, dùng paths từ đó (cho đánh giá thực tế)
    eval_folder = test_folder if test_folder else dataset_folder
    print(f"Đánh giá trên folder: {eval_folder}")

    # Tạo danh sách paths THEO ĐÚNG THỨ TỰ (nếu test_folder, giả định cấu trúc giống dataset)
    paths = []
    eval_labels = []  # Labels cho evaluation (nếu test, tự suy từ folder)
    for cls in sorted(os.listdir(eval_folder)):
        cls_path = os.path.join(eval_folder, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in sorted(os.listdir(cls_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                paths.append(os.path.join(cls_path, fname))
                eval_labels.append(cls)  # Suy labels từ subfolder

    if len(paths) != len(eval_labels):
        raise ValueError(f"Số ảnh ({len(paths)}) không khớp với số labels ({len(eval_labels)})")

    # Nếu không phải test_folder, check khớp với feats
    if not test_folder and len(paths) != len(feats):
        raise ValueError(f"Số ảnh ({len(paths)}) không khớp với số đặc trưng ({len(feats)})")

    print(f"Tổng số ảnh đánh giá: {len(paths)}")
    y_true = eval_labels  # Không dùng labels cũ nếu test_folder, và KHÔNG append duplicate
    y_pred = []

    for i, p in enumerate(paths):
        # DỰ ĐOÁN BẰNG NEAREST NEIGHBOR
        res, _ = recognize(p, features_file, labels_file, topk=1, threshold=threshold)
        pred_label = res[0][0] if res[0][0] != "Không nhận diện được" else "unknown"
        y_pred.append(pred_label)

        if (i + 1) % 20 == 0:
            print(f"Đã đánh giá {i + 1}/{len(paths)} ảnh")

    # Tính accuracy
    # unique_labels = sorted(set(y_true) | set(y_pred))  # Include 'unknown' nếu có
    labels_for_cm = sorted(unique_labels(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels_for_cm)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
    print(f"Độ chính xác: {acc*100:.2f}%")
    print("Ma trận nhầm lẫn:\n")
    print(cm)

    # Lưu model (nếu cần)
    model_data = {"features": feats, "labels": labels}
    os.makedirs("model", exist_ok=True)    
    print("\nĐã lưu mô hình tại: model/model.pkl")
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--test_dataset', required=False, help='Folder test riêng để đánh giá thực tế')
    parser.add_argument('--threshold', type=float, default=0.75, help='Ngưỡng similarity tối thiểu')
    args = parser.parse_args()
    evaluate(args.features, args.labels, args.dataset, test_folder=args.test_dataset, threshold=args.threshold)