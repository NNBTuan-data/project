# ĐỒ ÁN: HỆ THỐNG NHẬN DIỆN ĐỐI TƯỢNG DỰA TRÊN KẾT HỢP ĐẶC TRƯNG LBP, HOG, COLOR HISTOGRAM VÀ DEEP LEARNING

**Tên dự án:** Object Recognition System Using Combined LBP, HOG, Color Histogram Features and Deep Learning  
**Mã dự án:** CV-DL-2026  
**Thời gian thực hiện:** Tháng 01/2026  
**Nhóm:** Nhóm 8

---

## 👥 THÀNH VIÊN NHÓM VÀ PHÂN CÔNG

| STT | Họ Tên | Nhiệm Vụ Chính | Chi Tiết Công Việc |
|-----|--------|----------------|-------------------|
| 1 | **Thắng** | Classical ML + Báo Cáo 1 | - Dataset download & preprocessing<br>- Feature extraction (LBP, HOG, Color)<br>- SVM training & evaluation<br>- **Viết Word Report phần 1** (Giới thiệu, Classical ML) |
| 2 | **Định** | Deep Learning | - CNN architecture design<br>- Model builder (MobileNetV2)<br>- Data augmentation<br>- CNN training & optimization |
| 3 | **Bình** | GUI + Integration + Báo Cáo 2 | - GUI development (Tkinter)<br>- Model integration<br>- System testing<br>- **Viết Word Report phần 2** (Deep Learning, Kết quả, Demo) |
| 4 | **Tuấn** | Data & Features | - Dataset collection & cleaning<br>- Image preprocessing pipeline<br>- Feature engineering<br>- Code refactoring & documentation |

---

## 📋 PHÂN CÔNG CHI TIẾT

### 1. THẮNG - Classical Machine Learning + Báo Cáo 1

#### A. Công Việc Kỹ Thuật

**Dataset Download (15%):**
- File: `src/data/dataset_downloader.py`
- Download 100 ảnh/class từ DuckDuckGo (tổng ~800 ảnh)
- Xử lý lỗi, validate ảnh
- Multi-query search strategy (4 keywords/class)

> **💡 Note**: Thắng cũng **tự implement HOG algorithm từ scratch** (~300 LOC) thay vì dùng `skimage.hog` library để tăng technical complexity!

**Feature Extraction (55%):**
- `src/features/lbp_extractor.py` - LBP features (16,384 dims) - **CUSTOM**
- `src/features/hog_extractor.py` - **HOG features (8,100 dims) - CUSTOM (~300 LOC)**
  - Sobel gradient computation
  - Cell histogram với linear interpolation
  - L2-Hys block normalization
  - **100% tự implement, KHÔNG dùng `skimage.hog`!**
- `src/features/color_extractor.py` - Color histogram (96 dims)

**SVM Training (25%):**
- `src/models/svm_trainer.py`
- Feature combination với weights
- Train Linear SVM (C=5.0)
- Evaluation: accuracy, confusion matrix

**Scripts:**
- `scripts/1_download_dataset.py`
- `scripts/3b_train_svm.py`

#### B. Báo Cáo Word - Phần 1 (30-40 trang)

**Nội dung:**

**Chương 1: Giới Thiệu (5-7 trang)**
- 1.1. Đặt vấn đề
- 1.2. Mục tiêu dự án
- 1.3. Phạm vi nghiên cứu
- 1.4. Cấu trúc báo cáo

**Chương 2: Cơ Sở Lý Thuyết - Classical ML (10-12 trang)**
- 2.1. Local Binary Pattern (LBP)
  - Công thức toán học: `LBP(xc,yc) = Σ s(gi - gc) × 2^i`
  - **Thuật toán tự implement** (không dùng skimage)
  - Spatial histogram 4×4 cells
- 2.2. **Histogram of Oriented Gradients (HOG) - TỰ IMPLEMENT**
  - **Sobel operators** cho gradient computation (Gx, Gy)
  - **Magnitude & Orientation** calculation
  - **Cell histogram** với 9 bins và linear interpolation
  - **L2-Hysteresis block normalization**
  - **Implementation từ scratch** (~300 LOC)
- 2.3. Color Histogram
  - RGB color space
  - Histogram construction (OpenCV)
- 2.4. Support Vector Machine (SVM)
  - Linear kernel
  - Regularization parameter C=5.0
  - Training algorithm

**Chương 3: Dataset (5-7 trang)**
- 3.1. Thu thập dữ liệu (DuckDuckGo API)
- 3.2. Preprocessing pipeline
- 3.3. Phân bố 8 classes
- 3.4. Train/validation split (80/20)

**Chương 4: Implementation - Classical ML (8-10 trang)**
- 4.1. **Custom HOG Implementation** (chi tiết thuật toán)
  - Sobel gradient computation code
  - Cell histogram với linear interpolation
  - L2-Hys normalization method
- 4.2. Feature extraction pipeline
- 4.3. Feature combination strategy (weights: Color 0.2, LBP 0.3, HOG 0.5)
- 4.4. SVM training process
- 4.5. Hyperparameter tuning
- 4.6. Kết quả: Train 99.7%, Test 56.55% (overfitting analysis)

**Deliverables:**
- ✅ File Word: `BaoCao_Phan1_Thang.docx`
- ✅ Code: Dataset, Features (bao gồm **Custom HOG 300 LOC**), SVM modules
- ✅ **Technical Achievement**: Tự implement HOG algorithm từ scratch!

---

### 2. ĐỊNH - Deep Learning (CNN)

#### A. Công Việc Kỹ Thuật

**Model Architecture (40%):**
- File: `src/models/model_builder.py`
- Implement 3 architectures:
  - MobileNetV2 (main) - 3.5M params
  - ResNet50 (backup) - 24M params
  - Custom CNN (learning)

**Data Augmentation (20%):**
- File: `src/preprocessing/data_augmenter.py`
- ImageDataGenerator với:
  - Rotation: ±30°
  - Shift: 30%
  - Zoom: 25%
  - Brightness: [0.8, 1.2]
  - Horizontal flip

**Training Pipeline (30%):**
- File: `src/models/cnn_trainer.py`
- 2-phase training:
  - Phase 1: Freeze base, train top (50 epochs)
  - Phase 2: Fine-tune 30 layers (10 epochs)
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

**Optimization (10%):**
- Learning rate: 0.0001
- Batch size: 32
- Validation split: 20%

**Scripts:**
- `scripts/3a_train_cnn.py`

**Deliverables:**
- ✅ Code: Model builder, CNN trainer, Augmenter
- ✅ Trained model: `model/cnn_model.h5` (25 MB)
- ✅ **Accuracy: 81.33% validation** (Best epoch: 23)

---

### 3. BÌNH - GUI + Integration + Báo Cáo 2

#### A. Công Việc Kỹ Thuật

**GUI Development (40%):**
- File: `src/gui/main_window.py`
- Layout: 3 panels (Input | Results | Top-K)
- Features:
  - File browser
  - Drag & drop
  - Model selection (SVM/CNN)
  - Real-time prediction

**UI Components (20%):**
- File: `src/gui/ui_components.py`
- Components:
  - ImagePanel - Display images
  - ResultTextPanel - Show predictions
  - ThumbnailGrid - Top results
  - InfoLabel - Metadata

**Styling (10%):**
- File: `src/gui/ui_styles.py`
- Modern UI design
- Color scheme, fonts
- Professional look

**Integration (20%):**
- Load both models (SVM + CNN)
- Switch between models
- Confidence scores
- Error handling

**Testing (10%):**
- Unit tests
- Integration tests
- User testing

**Scripts:**
- `scripts/6_run_gui.py`

#### B. Báo Cáo Word - Phần 2 (30-40 trang)

**Nội dung:**

**Chương 5: Deep Learning (12-15 trang)**
- 5.1. Convolutional Neural Networks
  - Conv layers, Pooling, Activation
- 5.2. Transfer Learning
  - ImageNet pretrained
  - Fine-tuning strategy
- 5.3. MobileNetV2 Architecture
  - Depthwise separable convolutions
  - Inverted residuals
- 5.4. Data Augmentation
  - Techniques & impact

**Chương 6: Implementation - CNN (8-10 trang)**
- 6.1. Model building process
- 6.2. Training pipeline
- 6.3. Callbacks & optimization
- 6.4. Hyperparameters tuning

**Chương 7: Kết Quả & Đánh Giá (8-10 trang)**
- 7.1. So sánh Classical ML (56.55%) vs CNN (81.33%)
- 7.2. Confusion matrix analysis
- 7.3. Per-class accuracy
- 7.4. Training curves
- 7.5. Error analysis

**Chương 8: GUI & Demo (5-7 trang)**
- 8.1. Thiết kế giao diện
- 8.2. Tính năng chính
- 8.3. Screenshots demo
- 8.4. User guide

**Chương 9: Kết Luận (3-5 trang)**
- 9.1. Tóm tắt kết quả
- 9.2. Đóng góp của dự án
- 9.3. Hạn chế
- 9.4. Hướng phát triển tương lai

**Deliverables:**
- ✅ File Word: `BaoCao_Phan2_Binh.docx`
- ✅ Code: GUI modules
- ✅ Screenshots, demo video

---

### 4. TUẤN - Data Processing & Documentation

#### A. Công Việc Kỹ Thuật

**Image Preprocessing (30%):**
- File: `src/preprocessing/image_processor.py`
- Pipeline:
  - Read image (OpenCV)
  - Resize to 128×128
  - Grayscale conversion
  - Gaussian blur
  - Histogram equalization

**Feature Management (30%):**
- File: `src/features/feature_manager.py`
- Orchestrate feature extraction
- Feature combination
- Save/load .npy files
- Normalization

**Code Refactoring (20%):**
- Config: `src/config.py`
- Dataset loader: `src/data/dataset_loader.py`
- Modular structure (6 packages, 20+ modules)
- Type hints, docstrings
- Logging implementation

**Documentation (20%):**
- `README.md` - Installation & usage guide
- `requirements.txt` - All dependencies
- Code comments
- API documentation

**Scripts:**
- `scripts/2_extract_features.py`

**Deliverables:**
- ✅ Code: Preprocessing, Config, Feature manager
- ✅ Documentation: README, requirements
- ✅ Code quality: Clean, documented code

---

## 📊 KẾT QUẢ TỔNG HỢP

### Accuracy Comparison

| Model | Train Acc | Val Acc | Improvement |
|-------|-----------|---------|-------------|
| SVM (Classical) | 99.7% | 56.55% | Baseline |
| **CNN (MobileNetV2)** | ~95% | **81.33%** | **+24.78%** |

### Performance Metrics

| Metric | SVM | CNN |
|--------|-----|-----|
| Training Time | 6 min | 45 min |
| Inference Speed | 50ms | 30ms |
| Model Size | 3 MB | 25 MB |
| Top-3 Accuracy | - | 95% |

---

## 🗂️ CẤU TRÚC DỰ ÁN

```
project/
├── src/
│   ├── config.py                    # Tuấn - Configuration
│   ├── data/
│   │   ├── dataset_loader.py       # Tuấn
│   │   └── dataset_downloader.py   # Thắng
│   ├── preprocessing/
│   │   ├── image_processor.py      # Tuấn
│   │   └── data_augmenter.py       # Định
│   ├── features/
│   │   ├── feature_manager.py      # Tuấn
│   │   ├── color_extractor.py      # Thắng
│   │   ├── lbp_extractor.py        # Thắng
│   │   └── hog_extractor.py        # Thắng
│   ├── models/
│   │   ├── svm_trainer.py          # Thắng
│   │   ├── model_builder.py        # Định
│   │   └── cnn_trainer.py          # Định
│   └── gui/
│       ├── main_window.py          # Bình
│       ├── ui_components.py        # Bình
│       └── ui_styles.py            # Bình
├── scripts/
│   ├── 1_download_dataset.py       # Thắng
│   ├── 2_extract_features.py       # Tuấn
│   ├── 3a_train_cnn.py            # Định
│   ├── 3b_train_svm.py            # Thắng
│   └── 6_run_gui.py               # Bình
├── model/
│   ├── model.pkl                   # SVM model
│   └── cnn_model.h5               # CNN model
├── dataset/                        # 8 classes
├── features/                       # .npy files
├── README.md                       # Tuấn
├── requirements.txt                # Tuấn
└── PROJECT_REPORT.md              # This file
```

---

## ⏱️ TIMELINE (3 Tuần)

**Tuần 1: Data & Classical ML**
- Ngày 1-2: Setup + Download dataset (Thắng, Tuấn)
- Ngày 3-4: Feature extraction (Thắng, Tuấn)
- Ngày 5-7: SVM training (Thắng)

**Tuần 2: Deep Learning**
- Ngày 8-10: CNN architecture (Định)
- Ngày 11-13: Training + optimization (Định)
- Ngày 14: Evaluation & comparison (All)

**Tuần 3: GUI + Reports**
- Ngày 15-17: GUI development (Bình)
- Ngày 18-19: Integration + testing (Bình, Tuấn)
- Ngày 20-21: Word reports (Thắng, Bình)

---

## ✅ CHECKLIST

### Thắng
- [ ] Dataset downloader
- [ ] LBP/HOG/Color extractors
- [ ] SVM trainer
- [ ] Model evaluation
- [ ] **Word Report Part 1** (Chương 1-4)

### Định
- [ ] Model builder (3 architectures)
- [ ] Data augmentation
- [ ] CNN trainer (2-phase)
- [ ] Achieve >80% accuracy

### Bình
- [ ] GUI main window
- [ ] UI components & styling
- [ ] Model integration
- [ ] Testing
- [ ] **Word Report Part 2** (Chương 5-9)

### Tuấn
- [ ] Image preprocessing
- [ ] Feature manager
- [ ] Code refactoring
- [ ] README + requirements

---

## 📚 DELIVERABLES

### Code
- ✅ 6 packages, 20+ modules
- ✅ Type hints, docstrings
- ✅ Logging, error handling
- ✅ **Custom HOG implementation** (~300 LOC - không dùng skimage)
- ✅ **Custom LBP implementation** (~173 LOC)

### Models
- ✅ `model/model.pkl` (SVM - 56.55% accuracy)
- ✅ `model/cnn_model.h5` (CNN - 81.33% accuracy)

### Reports
- ✅ `BaoCao_Phan1_Thang.docx` (30-40 trang)
  - Bao gồm chi tiết **Custom HOG algorithm**
- ✅ `BaoCao_Phan2_Binh.docx` (30-40 trang)
- **Tổng: 60-80 trang**

### Presentation
- ✅ PowerPoint (20-30 slides)
- ✅ Demo video (5-10 phút)

### 🎯 Technical Highlights
- **Custom Algorithms**: HOG (300 LOC) + LBP (173 LOC)
- **No black-box libraries** for main features
- **Research-level implementation** từ papers gốc

---

**Ngày:** 06/01/2026  
**Version:** 3.0.0  
**Status:** ✅ Complete  

© 2026 Nhóm 8 - Object Recognition System
