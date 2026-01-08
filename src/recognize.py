import numpy as np
from utils import read_image, extract_feature_vector
# from scipy.spatial.distance import cdist

# Hàm tính cosine similarity
def cosine_similarity_vector(qv, features):
    # features: (N, D), qv: (D,)
    qv_norm = np.linalg.norm(qv)
    feats_norm = np.linalg.norm(features, axis=1)
    # tránh chia cho 0
    denom = feats_norm * qv_norm
    denom[denom == 0] = 1e-12
    sims = np.dot(features, qv) / denom
    return sims

def recognize(img_path, feat_path, labels_path, size=(128,128), threshold=0.75, topk=5):
    """
    Trả về topk kết quả THỰC TẾ theo cosine similarity
    - Nếu sim top1 < threshold, set label top1 = "Không nhận diện được" nhưng giữ sim thực
    - Các top còn lại giữ sim thực (không gán 0%)
    """
    features = np.load(feat_path)
    labels = np.load(labels_path)

    # Trích xuất đặc trưng ảnh đầu vào (đúng chuẩn, không thay đổi)
    img = read_image(img_path, size=size)
    qv = extract_feature_vector(img)

    # Tính cosine similarity
    sims = cosine_similarity_vector(qv, features)
    sorted_idx = np.argsort(sims)[::-1][:topk]

    # Lấy topk kết quả
    results = []
    match_indices = []

    for i, idx in enumerate(sorted_idx):
        sim = float(sims[idx])
        label = labels[idx]
        if i == 0 and sim < threshold:
            label = "Không nhận diện được"  # Chỉ top1 nếu thấp
        results.append((label, sim, int(idx)))
        match_indices.append(int(idx))

    return results, match_indices 