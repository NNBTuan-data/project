# src/utils.py
import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder

def list_images(dataset_path):
    img_paths = []
    labels = []
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        files.sort(key=lambda x: x.lower())
        for f in files:
            img_paths.append(os.path.join(cls_path, f))
            labels.append(cls)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return img_paths, labels

def read_image(path, size=(128,128)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.equalizeHist(img)
    return img

def normalize_hist(hist):
    hist = np.array(hist, dtype=np.float32)
    s = hist.sum()
    if s > 0:
        hist = hist / s
    return hist

def lbp_image(img, P=8, R=1):
    h, w = img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    for y in range(1, h-1):
        for x in range(1, w-1):
            center = img[y,x]
            code = 0
            for i, (dy, dx) in enumerate(neighbors):
                neighbor = img[y+dy, x+dx]
                code |= (1 << i) if neighbor >= center else 0
            lbp[y,x] = code
    return lbp

def lbp_hist(img, P=8, R=1, grid_x=8, grid_y=8, bins=256):
    lbp = lbp_image(img, P, R)
    h, w = img.shape
    gx = np.linspace(0, w, grid_x+1, dtype=int)
    gy = np.linspace(0, h, grid_y+1, dtype=int)
    feats = []
    for i in range(grid_y):
        for j in range(grid_x):
            x0, x1 = gx[j], gx[j+1]
            y0, y1 = gy[i], gy[i+1]
            patch = lbp[y0:y1, x0:x1]
            hist, _ = np.histogram(patch.flatten(), bins=bins, range=(0, bins))
            hist = normalize_hist(hist)
            feats.append(hist)
    return np.concatenate(feats)

def hog_feature(img, pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9):
    feat = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block, block_norm='L2-Hys', feature_vector=True)
    return feat

def extract_color_histogram(path, bins=32):
    img = cv2.imread(path)
    if img is None:
        return np.zeros(3*bins, dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256]).flatten()
    hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256]).flatten()
    hist = np.concatenate([hist_r, hist_g, hist_b])
    hist = hist / (hist.sum() + 1e-8)
    return hist.astype(np.float32)

def extract_lbp_histogram(img, grid_x=8, grid_y=8, bins=256):
    return lbp_hist(img, grid_x=grid_x, grid_y=grid_y, bins=bins)

def extract_hog_feature(img, **kwargs):
    return hog_feature(img, **kwargs)

def extract_feature_vector(img, lbp_grid=(8,8), hog_params=None):
    if hog_params is None:
        hog_params = {"pixels_per_cell":(8,8), "cells_per_block":(2,2), "orientations":9}
    lbp_v = extract_lbp_histogram(img, *lbp_grid)
    hog_v = extract_hog_feature(img, **hog_params)
    lbp_v = lbp_v.astype(np.float32)
    hog_v = hog_v.astype(np.float32)
    if lbp_v.sum() > 0:
        lbp_v = lbp_v / (np.linalg.norm(lbp_v) + 1e-8)
    if hog_v.sum() > 0:
        hog_v = hog_v / (np.linalg.norm(hog_v) + 1e-8)
    return np.concatenate([lbp_v, hog_v])