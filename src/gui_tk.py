# src/gui_tk.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import numpy as np
import joblib
from utils import list_images, read_image, extract_color_histogram, extract_lbp_histogram, extract_hog_feature
from pathlib import Path
import platform
from skimage.color import rgb2gray

# ĐƯỜNG DẪN
BASE_DIR = Path(__file__).parent.parent
DATASET_FOLDER = BASE_DIR / "dataset"
COLOR_FILE = BASE_DIR / "features" / "colors.npy"
LBP_FILE = BASE_DIR / "features" / "lbp.npy"
HOG_FILE = BASE_DIR / "features" / "hog.npy"
LABELS_FILE = BASE_DIR / "features" / "labels.npy"
SVM_MODEL_PATH = BASE_DIR / "model" / "model.pkl"

# MÀU
BG_COLOR = "#f8f9fa"
CARD_COLOR = "#ffffff"
PRIMARY = "#4361ee"
SUCCESS = "#06d6a0"
DANGER = "#ef476f"
TEXT_DARK = "#2d3436"
TEXT_LIGHT = "#636e72"
BORDER = "#dee2e6"

class ImageSearchApp:
    def __init__(self, master):
        self.master = master
        self.master.title("NHẬN DIỆN ẢNH CHUYÊN NGHIỆP (NHÓM 8)")
        self.master.geometry("1351x715")
        self.master.configure(bg=BG_COLOR)
        self.master.resizable(False, False)

        self.current_image_path = None
        self.svm_model = None
        self.colors = None
        self.lbp_features = None
        self.hog_features = None
        self.labels = None
        self.paths = None
        self.class_names = self.get_class_names()

        self.setup_ui()
        self.auto_prepare_system()

    def get_class_names(self):
        classes = [d.name for d in DATASET_FOLDER.iterdir() if d.is_dir()]
        classes.sort()
        return classes

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Helvetica', 20, 'bold'), foreground=PRIMARY, background=BG_COLOR)
        style.configure('Header.TLabel', font=('Helvetica', 13, 'bold'), foreground=TEXT_DARK, background=CARD_COLOR)
        style.configure('TButton', font=('Helvetica', 12, 'bold'), padding=12)
        style.map('TButton', background=[('active', PRIMARY)], foreground=[('active', 'white')])

        # HEADER
        header = tk.Frame(self.master, bg=BG_COLOR)
        header.pack(fill='x', pady=(10, 15))
        ttk.Label(header, text="HỆ THỐNG NHẬN DIỆN ẢNH VÀ TÌM KIẾM ĐỐI TƯỢNG CHUYÊN NGHIỆP", style='Title.TLabel').pack()

        main = tk.Frame(self.master, bg=BG_COLOR)
        main.pack(expand=True, fill='both', padx=20, pady=10)
        main.grid_columnconfigure((0, 1, 2), weight=1, uniform="col")

        # CỘT 1: ẢNH ĐẦU VÀO
        col1 = tk.Frame(main, bg=CARD_COLOR, relief='flat', bd=1, highlightbackground=BORDER, highlightthickness=1)
        col1.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(col1, text="ẢNH ĐẦU VÀO", style='Header.TLabel').pack(pady=(15, 10))
        self.input_frame = tk.Frame(col1, bg="#f1f3f5", relief='sunken', bd=2, width=380, height=380)
        self.input_frame.pack(pady=10, padx=20)
        self.input_frame.pack_propagate(False)
        self.input_label = tk.Label(self.input_frame, text="Chưa chọn ảnh", font=('Helvetica', 11, 'italic'), fg=TEXT_LIGHT, bg="#f1f3f5")
        self.input_label.pack(expand=True)

        btn_frame1 = tk.Frame(col1, bg=CARD_COLOR)
        btn_frame1.pack(pady=15)
        ttk.Button(btn_frame1, text="LỰA CHỌN ẢNH", command=self.open_image).pack(pady=8, fill='x', padx=40)
        self.search_btn = ttk.Button(btn_frame1, text="NHẬN DIỆN ẢNH", command=self.smart_search, state='disabled')
        self.search_btn.pack(pady=8, fill='x', padx=40)

        # CỘT 2: KẾT QUẢ + TOP 5
        col2 = tk.Frame(main, bg=CARD_COLOR, relief='flat', bd=1, highlightbackground=BORDER, highlightthickness=1)
        col2.grid(row=0, column=1, sticky="nsew", padx=10)
        ttk.Label(col2, text="KẾT QUẢ TÌM KIẾM", style='Header.TLabel').pack(pady=(15, 10), anchor='w', padx=20)
        self.result_text = tk.Text(col2, font=("Consolas", 11), bg="#fdfdfd", fg=TEXT_DARK, relief='flat', bd=0,
                                  highlightbackground=BORDER, highlightthickness=1, wrap='word', spacing1=5, spacing3=5,
                                  padx=15, pady=15, height=10)
        self.result_text.pack(fill='both', expand=True, padx=20, pady=(0, 10))

        self.canvas = tk.Canvas(col2, bg="#f0f0f0", highlightthickness=0)
        self.frame = tk.Frame(self.canvas, bg="#f0f0f0")
        self.h_scrollbar = ttk.Scrollbar(col2, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set)
        self.canvas.pack(side=tk.TOP, fill="both", expand=True, padx=20, pady=(0, 10))
        self.h_scrollbar.pack(side=tk.BOTTOM, fill="x")
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        if platform.system() == "Darwin":
            self.canvas.bind("<Button-4>", lambda e: self.canvas.xview_scroll(-1, "units"))
            self.canvas.bind("<Button-5>", lambda e: self.canvas.xview_scroll(1, "units"))
        else:
            self.canvas.bind("<Button-4>", lambda e: self.canvas.xview_scroll(-1, "units"))
            self.canvas.bind("<Button-5>", lambda e: self.canvas.xview_scroll(1, "units"))
        self.frame.bind("<Configure>", self.update_scroll_region)

        self.scroll_guide = tk.Label(col2, text="Kéo thanh cuộn ngang hoặc lăn chuột để xem ảnh thứ 4.", 
                                   font=('Helvetica', 9), fg=TEXT_LIGHT, bg=CARD_COLOR)
        self.scroll_guide.pack(pady=(0, 5))

        # CỘT 3: ẢNH MATCH + DỰ ĐOÁN LỚP
        col3 = tk.Frame(main, bg=CARD_COLOR, relief='flat', bd=1, highlightbackground=BORDER, highlightthickness=1)
        col3.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        ttk.Label(col3, text="ẢNH TƯƠNG ĐỒNG NHẤT", style='Header.TLabel').pack(pady=(15, 10))

        self.match_frame = tk.Frame(col3, bg="#f1f3f5", relief='sunken', bd=2, width=380, height=380)
        self.match_frame.pack(pady=10, padx=20)
        self.match_frame.pack_propagate(False)
        self.match_label = tk.Label(self.match_frame, text="Chưa có kết quả", font=('Helvetica', 11, 'italic'), fg=TEXT_LIGHT, bg="#f1f3f5")
        self.match_label.pack(expand=True)

        self.top1_label = tk.Label(col3, text="", font=('Helvetica', 12, 'bold'), fg=PRIMARY, bg=CARD_COLOR)
        self.top1_label.pack(pady=(0, 8))

        self.predicted_label = tk.Label(col3, text="", font=('Helvetica', 14, 'bold'), fg=DANGER, bg=CARD_COLOR)
        self.predicted_label.pack(pady=(5, 8))

        footer = tk.Label(self.master, text="© 2025 LBP+HOG+Color Pro Recognition", font=('Helvetica', 9), fg=TEXT_LIGHT, bg=BG_COLOR)
        footer.pack(side='bottom', pady=12)

    def _on_mousewheel(self, event):
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def update_scroll_region(self, event):
        region = self.canvas.bbox("all")
        if region:
            self.canvas.configure(scrollregion=region)

    def log(self, text, tag="info"):
        if tag != "debug":
            self.result_text.insert(tk.END, text + "\n", tag)
            self.result_text.see(tk.END)
            self.master.update()

    def auto_prepare_system(self):
        self.log("KHỞI TẠO HỆ THỐNG...", "header")
        if not DATASET_FOLDER.exists():
            messagebox.showerror("Lỗi", f"Không tìm thấy: {DATASET_FOLDER}")
            return

        if not all(f.exists() for f in [COLOR_FILE, LBP_FILE, HOG_FILE, LABELS_FILE]):
            self.log("TRÍCH XUẤT ĐẶC TRƯNG...", "process")
            from src.extract_features import main as extract
            extract(type("Args", (), {"dataset": "dataset", "size": 128}))

        if not SVM_MODEL_PATH.exists():
            self.log("HUẤN LUYỆN MÔ HÌNH...", "process")
            from src.train import main as train
            train()

        self.load_all_data()

    def load_all_data(self):
        try:
            self.colors = np.load(COLOR_FILE)
            self.lbp_features = np.load(LBP_FILE)
            self.hog_features = np.load(HOG_FILE)
            self.labels = np.load(LABELS_FILE)
            if SVM_MODEL_PATH.exists():
                self.svm_model = joblib.load(SVM_MODEL_PATH)
            self.paths, _ = list_images(DATASET_FOLDER)
            self.log(f"HOÀN TẤT: {len(self.labels)} ảnh", "success")
        except Exception as e:
            self.log(f"LỖI TẢI: {e}", "error")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.current_image_path = path
        try:
            img = Image.open(path).convert("RGB")
            display_img = img.copy(); display_img.thumbnail((360, 360))
            imgtk = ImageTk.PhotoImage(display_img)
            self.input_label.config(image=imgtk, text=""); self.input_label.image = imgtk
            self.result_text.delete("1.0", tk.END)
            self.search_btn.config(state='normal')
            self.clear_results()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không mở ảnh:\n{e}")

    def clear_results(self):
        self.match_label.config(image='', text="Chưa có kết quả")
        self.top1_label.config(text="")
        self.predicted_label.config(text="")
        for widget in self.frame.winfo_children():
            widget.destroy()

    def cosine_similarity(self, qv, db):
        qv_norm = np.linalg.norm(qv)
        if qv_norm == 0: return np.zeros(len(db))
        qv = qv / qv_norm
        db_norms = np.linalg.norm(db, axis=1)
        db_norms[db_norms == 0] = 1e-8
        db = db / db_norms[:, np.newaxis]
        return np.dot(db, qv)

    # def smart_search(self):
    #     if not self.current_image_path or not self.svm_model:
    #         return
        
    #     self.result_text.delete("1.0", tk.END)
    #     self.log("TÌM KIẾM ẢNH TƯƠNG ĐỒNG NHẤT (TOÀN BỘ DATASET)...", "header")

    #     try:
    #         # === TRÍCH XUẤT ĐẶC TRƯNG ===
    #         img_pil = Image.open(self.current_image_path).convert("RGB").resize((128, 128))
    #         img_gray = rgb2gray(np.array(img_pil))
    #         query_color = extract_color_histogram(self.current_image_path)
    #         query_lbp   = extract_lbp_histogram(img_gray)
    #         query_hog   = extract_hog_feature(img_gray)

    #         # === DỰ ĐOÁN LỚP BỞI model.pkl ===
    #         X_query = np.hstack([query_color * 0.2, query_lbp * 0.3, query_hog * 0.5])
    #         pred_idx = self.svm_model.predict([X_query])[0]
    #         pred_class = self.class_names[pred_idx].upper()

    #         # === TÍNH ĐỘ TƯƠNG ĐỒNG TRÊN TOÀN BỘ DATASET ===
    #         sim_color = self.cosine_similarity(query_color, self.colors)
    #         sim_lbp   = self.cosine_similarity(query_lbp,   self.lbp_features)
    #         sim_hog   = self.cosine_similarity(query_hog,   self.hog_features)

    #         # TRỌNG SỐ TỐI ƯU: HOG 0.5, LBP 0.3, COLOR 0.2
    #         weighted = sim_color * 0.2 + sim_lbp * 0.3 + sim_hog * 0.5
    #         top5_idx = np.argsort(weighted)[::-1][:5]
    #         best_idx = top5_idx[0]

    #         # === HIỂN THỊ ẢNH TỐT NHẤT ===
    #         best_path = self.paths[best_idx]
    #         img_match = Image.open(best_path).convert("RGB").resize((360, 360))
    #         imgtk = ImageTk.PhotoImage(img_match)
    #         self.match_label.config(image=imgtk, text="")
    #         self.match_label.image = imgtk
    #         best_class = self.class_names[self.labels[best_idx]].upper()
    #         self.top1_label.config(text=f"{best_class} - {weighted[best_idx]*100:.1f}%")
    #         self.predicted_label.config(text=f"DỰ ĐOÁN LỚP: {pred_class}")

    #         # === TOP 4 ẢNH (2-5) ===
    #         top4_paths = [self.paths[i] for i in top5_idx[1:5]]
    #         top4_scores = [weighted[i] for i in top5_idx[1:5]]
    #         self.show_top5(top4_paths, top4_scores)

    #         # === LOG KẾT QUẢ (KHỚP 100%) ===
    #         self.log(f"\nDỰ ĐOÁN LỚP (model.pkl): {pred_class}", "header_svm")
    #         self.log("\nTOP 5 ẢNH TƯƠNG ĐỒNG NHẤT (TOÀN BỘ DATASET)", "header")
    #         self.log("═" * 70, "line")
    #         self.log("→ Dùng COLOR 0.2 + LBP 0.3 + HOG 0.5 + model.pkl", "note")
    #         for i, idx in enumerate(top5_idx, 1):
    #             sim = weighted[idx] * 100
    #             cls = self.class_names[self.labels[idx]].upper()
    #             self.log(f"{i}. {cls:15}: {sim:6.2f}%", "final")
    #         self.log("═" * 70, "line")

    #     except Exception as e:
    #         import traceback
    #         self.log(f"LỖI: {traceback.format_exc()}", "error")

    #     # ĐỊNH DẠNG
    #     self.result_text.tag_config("header", foreground=PRIMARY, font=("Consolas", 13, "bold"))
    #     self.result_text.tag_config("header_svm", foreground=SUCCESS, font=("Consolas", 12, "bold"))
    #     self.result_text.tag_config("final", foreground=SUCCESS, font=("Consolas", 14, "bold"))
    #     self.result_text.tag_config("note", foreground="#666", font=("Consolas", 10, "italic"))
    #     self.result_text.tag_config("line", foreground="#b2bec3")
    #     self.result_text.tag_config("error", foreground=DANGER)
    
#*********************************ƯU TIÊN NHÃN DỰA VÀO CÁC ĐẶC TRƯNG************************
    def smart_search(self):
        if not self.current_image_path or not self.svm_model:
            return
        
        self.result_text.delete("1.0", tk.END)
        self.clear_results()
        self.log("TÌM KIẾM ẢNH TƯƠNG ĐỒNG NHẤT (TOÀN BỘ DATASET)...", "header")

        try:
            # === TRÍCH XUẤT ĐẶC TRƯNG ===
            img_pil = Image.open(self.current_image_path).convert("RGB").resize((128, 128))
            img_gray = rgb2gray(np.array(img_pil))
            query_color = extract_color_histogram(self.current_image_path)
            query_lbp   = extract_lbp_histogram(img_gray)
            query_hog   = extract_hog_feature(img_gray)

            # === DỰ ĐOÁN LỚP BỞI model.pkl ===
            X_query = np.hstack([query_color * 0.2, query_lbp * 0.3, query_hog * 0.5])
            pred_idx = self.svm_model.predict([X_query])[0]
            pred_class_model = self.class_names[pred_idx].upper()

            # === TÍNH ĐỘ TƯƠNG ĐỒNG TRÊN TOÀN BỘ DATASET ===
            sim_color = self.cosine_similarity(query_color, self.colors)
            sim_lbp   = self.cosine_similarity(query_lbp,   self.lbp_features)
            sim_hog   = self.cosine_similarity(query_hog,   self.hog_features)

            # TRỌNG SỐ TỐI ƯU: HOG 0.5, LBP 0.3, COLOR 0.2
            weighted = sim_color * 0.2 + sim_lbp * 0.3 + sim_hog * 0.5
            top5_idx = np.argsort(weighted)[::-1][:5]
            best_idx = top5_idx[0]

            # === NHÃN TỪ TOP 1 (ẢNH GIỐNG NHẤT) ===
            best_class_top1 = self.class_names[self.labels[best_idx]].upper()
            top1_sim = weighted[best_idx] * 100

            # === SO SÁNH & CHỌN NHÃN CHÍNH XÁC NHẤT ===
            if pred_class_model == best_class_top1:
                final_pred_class = pred_class_model
                self.log(f"→ Model & Top 1 ĐỒNG THUẬN: {final_pred_class}", "note")
            else:
                final_pred_class = best_class_top1  # Ưu tiên Top 1
                self.log(f"→ Model: {pred_class_model} | Top 1: {best_class_top1}", "note")
                self.log(f"→ ƯU TIÊN NHÃN TOP 1 (chính xác hơn): {final_pred_class}", "final")

            # === HIỂN THỊ ẢNH TỐT NHẤT ===
            best_path = self.paths[best_idx]
            img_match = Image.open(best_path).convert("RGB").resize((360, 360))
            imgtk = ImageTk.PhotoImage(img_match)
            self.match_label.config(image=imgtk, text="")
            self.match_label.image = imgtk
            self.top1_label.config(text=f"{best_class_top1} - {top1_sim:.1f}%")
            self.predicted_label.config(text=f"DỰ ĐOÁN: {final_pred_class}")

            # === TOP 4 ẢNH (2-5) ===
            top4_paths = [self.paths[i] for i in top5_idx[1:5]]
            top4_scores = [weighted[i] for i in top5_idx[1:5]]
            self.show_top5(top4_paths, top4_scores)

            # === LOG KẾT QUẢ ===
            self.log(f"\nDỰ ĐOÁN LỚP (model.pkl): {pred_class_model}", "header_svm")
            self.log(f"TOP 1 GIỐNG NHẤT: {best_class_top1} ({top1_sim:.1f}%)", "header_svm")
            self.log("\nTOP 5 ẢNH TƯƠNG ĐỒNG NHẤT (TOÀN BỘ DATASET)", "header")
            self.log("═" * 70, "line")
            self.log("→ Dùng COLOR 0.2 + LBP 0.3 + HOG 0.5 + model.pkl", "note")
            for i, idx in enumerate(top5_idx, 1):
                sim = weighted[idx] * 100
                cls = self.class_names[self.labels[idx]].upper()
                self.log(f"{i}. {cls:15}: {sim:6.2f}%", "final")
            self.log("═" * 70, "line")

        except Exception as e:
            import traceback
            self.log(f"LỖI: {traceback.format_exc()}", "error")

        # ĐỊNH DẠNG
        self.result_text.tag_config("header", foreground=PRIMARY, font=("Consolas", 13, "bold"))
        self.result_text.tag_config("header_svm", foreground=SUCCESS, font=("Consolas", 12, "bold"))
        self.result_text.tag_config("final", foreground=SUCCESS, font=("Consolas", 14, "bold"))
        self.result_text.tag_config("note", foreground="#666", font=("Consolas", 10, "italic"))
        self.result_text.tag_config("line", foreground="#b2bec3")
        self.result_text.tag_config("error", foreground=DANGER)

    def show_top5(self, paths, scores):
        for widget in self.frame.winfo_children():
            widget.destroy()

        title = tk.Label(self.frame, text="TOP 4 ẢNH TƯƠNG TỰ (2-5)", 
                        font=('Helvetica', 12, 'bold'), fg=PRIMARY, bg=CARD_COLOR)
        title.pack(anchor='w', pady=(5, 8))

        note = tk.Label(self.frame, text="→ Độ tương tự khớp 100% với bảng kết quả",
                       font=('Helvetica', 9, 'italic'), fg=TEXT_LIGHT, bg=CARD_COLOR)
        note.pack(anchor='w', padx=10)

        row = tk.Frame(self.frame, bg="#e0e0e0", relief='raised', bd=1)
        row.pack(fill='x', padx=10, pady=5)

        for path, sim in zip(paths, scores):
            try:
                img = Image.open(path).convert("RGB").resize((90, 90), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(img)
                col = tk.Frame(row, bg="#d0d0d0", width=110, height=130, highlightbackground=BORDER, highlightthickness=2)
                col.pack(side='left', padx=10, pady=5)
                col.pack_propagate(False)
                lbl_img = tk.Label(col, image=imgtk, bg=CARD_COLOR, relief='solid', bd=1)
                lbl_img.image = imgtk
                lbl_img.pack(pady=(8, 3))
                color = SUCCESS if sim*100 >= 90 else ("#f39c12" if sim*100 >= 80 else TEXT_DARK)
                tk.Label(col, text=f"{sim*100:.1f}%", font=('Consolas', 10, 'bold'), fg=color, bg=CARD_COLOR).pack()
            except Exception as e:
                print(f"[TOP4] Lỗi: {e}")
                continue

        self.master.after(100, self.update_scroll_region, None)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()