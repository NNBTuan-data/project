"""
Main Window - Giao diện chính của ứng dụng
Refactored từ gui_tk.py với components và styles tách riêng
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import logging
import numpy as np
import joblib
from pathlib import Path

from ..config import (DATASET_DIR, COLOR_FEATURES_FILE, LBP_FEATURES_FILE,
                      HOG_FEATURES_FILE, LABELS_FILE, SVM_MODEL_PATH, IMAGE_SIZE,
                      CNN_MODEL_PATH)  # Thêm CNN_MODEL_PATH
from ..data.dataset_loader import DatasetLoader
from ..preprocessing.image_processor import ImageProcessor
from ..features import ColorExtractor, LBPExtractor, HOGExtractor
from .ui_styles import *
from .ui_components import ImagePanel, ResultTextPanel, ThumbnailGrid, InfoLabel

logger = logging.getLogger(__name__)


class MainWindow:
    """
    Cửa sổ chính của ứng dụng nhận diện ảnh
    """
    
    def __init__(self, master):
        """
        Khởi tạo Main Window
        
        Args:
            master: Tk root window
        """
        self.master = master
        self.master.title("NHẬN DIỆN ẢNH CHUYÊN NGHIỆP (NHÓM 2)")
        self.master.geometry(WINDOW_SIZE)
        self.master.configure(bg=BG_COLOR)
        # Allow resizing
        self.master.resizable(True, True)
        
        # Auto maximize
        try:
            self.master.state('zoomed')
        except:
            pass
        
        # State variables
        self.current_image_path = None
        self.svm_model = None
        self.cnn_model = None  # Thêm CNN model
        self.colors = None
        self.lbp_features = None
        self.hog_features = None
        self.labels = None
        self.paths = None
        self.class_names = None
        
        # Setup UI
        self.setup_ui()
        
        # Auto prepare system
        self.auto_prepare_system()
        
        logger.info("MainWindow initialized")
    
    def setup_ui(self):
        """Thiết lập giao diện"""
        # Configure ttk styles
        style = ttk.Style()
        configure_ttk_styles(style)
        
        # Header
        header = tk.Frame(self.master, bg=BG_COLOR)
        header.pack(fill='x', pady=(10, 15))
        ttk.Label(header, 
                 text="HỆ THỐNG NHẬN DIỆN ẢNH VÀ TÌM KIẾM ĐỐI TƯỢNG CHUYÊN NGHIỆP",
                 style='Title.TLabel').pack()
        
        # Main container
        main = tk.Frame(self.master, bg=BG_COLOR)
        main.pack(expand=True, fill='both', padx=20, pady=10)
        main.grid_columnconfigure((0, 1, 2), weight=1, uniform="col")
        
        # Column 1: Input Image
        self.input_panel = ImagePanel(main, "ẢNH ĐẦU VÀO")
        self.input_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Buttons for input
        btn_frame1 = tk.Frame(self.input_panel, bg=CARD_COLOR)
        btn_frame1.pack(pady=15)
        
        ttk.Button(btn_frame1, text="LỰA CHỌN ẢNH",
                  command=self.open_image).pack(pady=8, fill='x', padx=40)
        
        self.search_btn = ttk.Button(btn_frame1, text="NHẬN DIỆN ẢNH",
                                    command=self.smart_search, state='disabled')
        self.search_btn.pack(pady=8, fill='x', padx=40)
        
        # Column 2: Results - CHỈ HIỂN THỊ KẾT QUẢ TEXT
        col2 = tk.Frame(main, bg=CARD_COLOR, relief='flat', bd=2,
                       highlightbackground=BORDER, highlightthickness=2)
        col2.grid(row=0, column=1, sticky="nsew", padx=15)
        
        # Result text panel - FULL HEIGHT
        self.result_panel = ResultTextPanel(col2, height=25)
        self.result_panel.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Column 3: Best Match - TO HƠN VÀ ĐẸP HƠN
        self.match_panel = ImagePanel(main, "🎯 ẢNH TƯƠNG ĐỒNG NHẤT", width=520, height=520)
        self.match_panel.grid(row=0, column=2, sticky="nsew", padx=(15, 0))
        
        # Info labels với style mới
        self.top1_label = InfoLabel(self.match_panel)
        self.top1_label.config(font=('Segoe UI', 16, 'bold'), fg=ACCENT)
        self.top1_label.pack(pady=(10, 5))
        
        self.predicted_label = InfoLabel(self.match_panel)
        self.predicted_label.config(font=('Segoe UI', 18, 'bold'), fg=SUCCESS)
        self.predicted_label.pack(pady=(5, 15))
        
        # Footer với style mới
        footer = tk.Label(self.master, text="💎 © 2025 Advanced AI Image Recognition System | Powered by CNN & SVM",
                         font=FONT_SMALL, fg=TEXT_LIGHT, bg=BG_COLOR)
        footer.pack(side='bottom', pady=15)
    
    def get_class_names(self):
        """Lấy danh sách class names từ dataset"""
        try:
            loader = DatasetLoader(DATASET_DIR)
            return loader.get_class_names()
        except:
            logger.warning("Không lấy được class names từ dataset")
            return []
    
    def auto_prepare_system(self):
        """Tự động chuẩn bị hệ thống"""
        self.result_panel.log("KHỞI TẠO HỆ THỐNG...", "header")
        
        if not DATASET_DIR.exists():
            messagebox.showerror("Lỗi", f"Không tìm thấy: {DATASET_DIR}")
            return
        
        # Auto extract features if not exists
        if not all(f.exists() for f in [COLOR_FEATURES_FILE, LBP_FEATURES_FILE,
                                         HOG_FEATURES_FILE, LABELS_FILE]):
            self.result_panel.log("TRÍCH XUẤT ĐẶC TRƯNG...", "process")
            try:
                from ..features.feature_manager import FeatureManager
                manager = FeatureManager()
                manager.extract_all_features(show_progress=False)
            except Exception as e:
                logger.error(f"Lỗi trích xuất features: {e}")
        
        # Auto train if model not exists
        if not SVM_MODEL_PATH.exists():
            self.result_panel.log("HUẤN LUYỆN MÔ HÌNH...", "process")
            try:
                from ..models.svm_trainer import SVMTrainer
                trainer = SVMTrainer()
                trainer.train()
            except Exception as e:
                logger.error(f"Lỗi train model: {e}")
        
        # Load all data
        self.load_all_data()
    
    def load_all_data(self):
        """Load features và models vào RAM"""
        try:
            self.colors = np.load(str(COLOR_FEATURES_FILE))
            self.lbp_features = np.load(str(LBP_FEATURES_FILE))
            self.hog_features = np.load(str(HOG_FEATURES_FILE))
            self.labels = np.load(str(LABELS_FILE))
            
            # Load SVM model
            if SVM_MODEL_PATH.exists():
                self.svm_model = joblib.load(str(SVM_MODEL_PATH))
                logger.info("SVM model loaded")
            
            # Load CNN model (ƯU TIÊN)
            if CNN_MODEL_PATH.exists():
                from tensorflow.keras.models import load_model
                self.cnn_model = load_model(str(CNN_MODEL_PATH))
                logger.info("CNN model loaded ✓")
            
            loader = DatasetLoader(DATASET_DIR)
            self.paths, _ = loader.load_dataset()
            self.class_names = loader.get_class_names()
            
            model_info = []
            if self.cnn_model:
                model_info.append("CNN ✓")
            if self.svm_model:
                model_info.append("SVM ✓")
            
            self.result_panel.log(f"✅ HOÀN TẤT: {len(self.labels)} ảnh | Models: {', '.join(model_info)}\n", "final")
            logger.info(f"Loaded {len(self.labels)} images")
            
        except Exception as e:
            self.result_panel.log(f"❌ LỖI TẢI: {e}", "error")
            logger.error(f"Error loading data: {e}")
    
    def open_image(self):
        """Mở dialog chọn ảnh"""
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        
        if not path:
            return
        
        self.current_image_path = path
        
        try:
            # Display image
            self.input_panel.set_image(path, resize=PREVIEW_IMAGE_SIZE)
            
            # Clear results
            self.result_panel.clear()
            self.search_btn.config(state='normal')
            self.clear_results()
            
            logger.info(f"Opened image: {path}")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không mở ảnh:\n{e}")
            logger.error(f"Error opening image: {e}")
    
    def clear_results(self):
        """Xóa kết quả cũ"""
        self.match_panel.clear()
        self.top1_label.set_text("")
        self.predicted_label.set_text("")
    
    def cosine_similarity(self, qv, db):
        """Tính cosine similarity"""
        qv_norm = np.linalg.norm(qv)
        if qv_norm == 0:
            return np.zeros(len(db))
        qv = qv / qv_norm
        
        db_norms = np.linalg.norm(db, axis=1)
        db_norms[db_norms == 0] = 1e-8
        db = db / db_norms[:, np.newaxis]
        
        return np.dot(db, qv)
    
    def smart_search(self):
        """Tìm kiếm và nhận diện ảnh - SỬ DỤNG CNN MODEL"""
        if not self.current_image_path:
            return
        
        self.result_panel.clear()
        self.clear_results()
        self.result_panel.log("🔍 NHẬN DIỆN ẢNH BẰNG CNN...\n", "header")
        
        try:
            # Nếu có CNN, dùng CNN (81% accuracy)
            if self.cnn_model:
                from tensorflow.keras.preprocessing import image as keras_image
                import tensorflow as tf
                
                # Preprocess image cho CNN
                img = keras_image.load_img(self.current_image_path, target_size=IMAGE_SIZE)
                img_array = keras_image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalize
                
                # CNN prediction
                predictions = self.cnn_model.predict(img_array, verbose=0)[0]
                pred_idx = np.argmax(predictions)
                confidence = predictions[pred_idx] * 100
                pred_class = self.class_names[pred_idx].upper()
                
                self.result_panel.log(f"🤖 CNN MODEL PREDICTION:\n", "header")
                self.result_panel.log(f"   → {pred_class}\n", "final")
                self.result_panel.log(f"   → Confidence: {confidence:.2f}%\n\n", "header_svm")
                
                # Tìm ảnh tương đồng nhất bằng similarity
                processor = ImageProcessor(target_size=IMAGE_SIZE)
                img_gray = processor.read_image(self.current_image_path)
                
                color_ext = ColorExtractor()
                lbp_ext = LBPExtractor()
                hog_ext = HOGExtractor()
                
                query_color = color_ext.extract(self.current_image_path)
                query_lbp = lbp_ext.extract(img_gray)
                query_hog = hog_ext.extract(img_gray)
                
                # Calculate similarities
                sim_color = self.cosine_similarity(query_color, self.colors)
                sim_lbp = self.cosine_similarity(query_lbp, self.lbp_features)
                sim_hog = self.cosine_similarity(query_hog, self.hog_features)
                
                weighted = sim_color * 0.2 + sim_lbp * 0.3 + sim_hog * 0.5
                best_idx = np.argmax(weighted)
                
                best_class_sim = self.class_names[self.labels[best_idx]].upper()
                top1_sim = weighted[best_idx] * 100
                
                # Display kết quả
                self.match_panel.set_image(self.paths[best_idx], resize=PREVIEW_IMAGE_SIZE)
                self.top1_label.set_text(f"🎯 {best_class_sim} - {top1_sim:.1f}%")
                self.predicted_label.set_text(f"✨ CNN: {pred_class} ({confidence:.1f}%)")
                
                # Log results
                self.result_panel.log("═" * 65 + "\n", "line")
                self.result_panel.log(f"🎯 ẢNH TƯƠNG ĐỒNG NHẤT (Similarity):\n", "header")
                self.result_panel.log(f"   → {best_class_sim}\n", "final")
                self.result_panel.log(f"   → Độ tương đồng: {top1_sim:.2f}%\n\n", "header_svm")
                
                # So sánh CNN vs Similarity
                if pred_class == best_class_sim:
                    self.result_panel.log("✅ CNN & Similarity ĐỒNG THUẬN!\n\n", "header_svm")
                    final_result = pred_class
                else:
                    self.result_panel.log(f"⚠️ CNN: {pred_class} | Similarity: {best_class_sim}\n", "note")
                    self.result_panel.log(f"→ Ưu tiên kết quả CNN (Accuracy cao hơn)\n\n", "header_svm")
                    final_result = pred_class
                
                self.result_panel.log(f"🏆 KẾT QUẢ CUỐI CÙNG:\n", "header")
                self.result_panel.log(f"   ⭐ {final_result}\n", "final")
                
                self.result_panel.log("═" * 65 + "\n", "line")
                self.result_panel.log("\n💡 CNN Model: MobileNetV2 | Val Accuracy: 81.33%", "note")
                
            # Fallback to SVM nếu không có CNN
            elif self.svm_model:
                self.result_panel.log("⚠️ CNN model not found, using SVM...\n", "note")
                
                processor = ImageProcessor(target_size=IMAGE_SIZE)
                img_gray = processor.read_image(self.current_image_path)
                
                color_ext = ColorExtractor()
                lbp_ext = LBPExtractor()
                hog_ext = HOGExtractor()
                
                query_color = color_ext.extract(self.current_image_path)
                query_lbp = lbp_ext.extract(img_gray)
                query_hog = hog_ext.extract(img_gray)
                
                X_query = np.hstack([query_color * 0.2, query_lbp * 0.3, query_hog * 0.5])
                pred_idx = self.svm_model.predict([X_query])[0]
                pred_class = self.class_names[pred_idx].upper()
                
                weighted = self.cosine_similarity(query_color, self.colors) * 0.2 + \
                          self.cosine_similarity(query_lbp, self.lbp_features) * 0.3 + \
                          self.cosine_similarity(query_hog, self.hog_features) * 0.5
                best_idx = np.argmax(weighted)
                
                self.match_panel.set_image(self.paths[best_idx], resize=PREVIEW_IMAGE_SIZE)
                self.top1_label.set_text(f"SVM: {pred_class}")
                self.predicted_label.set_text(f"Accuracy: 56.55%")
            
            logger.info(f"Recognition completed")
            
        except Exception as e:
            import traceback
            self.result_panel.log(f"❌ LỖI:\n{traceback.format_exc()}", "error")
            logger.error(f"Error in smart_search: {e}", exc_info=True)


def run_gui():
    """Entry point để chạy GUI"""
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_gui()
