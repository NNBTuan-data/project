"""
Reusable UI Components
Các widget tái sử dụng cho GUI
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from typing import List, Tuple
import logging

from .ui_styles import *

logger = logging.getLogger(__name__)


class ImagePanel(tk.Frame):
    """
    Panel hiển thị ảnh với label
    """
    
    def __init__(self, parent, title: str, width: int = 380, height: int = 380, **kwargs):
        """
        Khởi tạo ImagePanel
        
        Args:
            parent: Parent widget
            title: Tiêu đề panel
            width, height: Kích thước frame
        """
        super().__init__(parent, bg=CARD_COLOR, relief='flat', bd=1,
                        highlightbackground=BORDER, highlightthickness=1, **kwargs)
        
        self.width = width
        self.height = height
        
        # Title
        ttk.Label(self, text=title, style='Header.TLabel').pack(pady=(15, 10))
        
        # Image frame
        self.img_frame = tk.Frame(self, bg="#f1f3f5", relief='sunken', bd=2,
                                 width=width, height=height)
        self.img_frame.pack(pady=10, padx=20)
        self.img_frame.pack_propagate(False)
        
        # Image label
        self.img_label = tk.Label(self.img_frame, text="Chưa có ảnh",
                                 font=FONT_ITALIC, fg=TEXT_LIGHT, bg="#f1f3f5")
        self.img_label.pack(expand=True)
    
    def set_image(self, img_path: str, resize: Tuple[int, int] = None):
        """
        Hiển thị ảnh
        
        Args:
            img_path: Đường dẫn ảnh
            resize: Kích thước resize (width, height)
        """
        try:
            img = Image.open(img_path).convert("RGB")
            
            if resize:
                img.thumbnail(resize)
            else:
                img.thumbnail((self.width - 20, self.height - 20))
            
            imgtk = ImageTk.PhotoImage(img)
            self.img_label.config(image=imgtk, text="")
            self.img_label.image = imgtk  # Keep reference
            
        except Exception as e:
            logger.error(f"Không hiển thị được ảnh: {e}")
            self.img_label.config(text=f"Lỗi: {e}")
    
    def clear(self):
        """Xóa ảnh"""
        self.img_label.config(image='', text="Chưa có ảnh")


class ResultTextPanel(tk.Frame):
    """
    Panel hiển thị text kết quả với scrollbar
    """
    
    def __init__(self, parent, height: int = 10, **kwargs):
        super().__init__(parent, bg=CARD_COLOR, **kwargs)
        
        # Title
        ttk.Label(self, text="KẾT QUẢ TÌM KIẾM", style='Header.TLabel').pack(
            pady=(15, 10), anchor='w', padx=20)
        
        # Text widget
        self.text = tk.Text(self, font=FONT_TEXT, bg="#fdfdfd", fg=TEXT_DARK,
                          relief='flat', bd=0, highlightbackground=BORDER,
                          highlightthickness=1, wrap='word', spacing1=5, spacing3=5,
                          padx=15, pady=15, height=height)
        self.text.pack(fill='both', expand=True, padx=20, pady=(0, 10))
        
        # Configure tags
        for tag_name, tag_config in TEXT_TAGS.items():
            self.text.tag_config(tag_name, **tag_config)
    
    def log(self, message: str, tag: str = "info"):
        """
        Thêm message vào text
        
        Args:
            message: Nội dung
            tag: Tag để format
        """
        if tag != "debug":
            self.text.insert(tk.END, message + "\n", tag)
            self.text.see(tk.END)
            self.winfo_toplevel().update()
    
    def clear(self):
        """Xóa toàn bộ text"""
        self.text.delete("1.0", tk.END)


class ThumbnailGrid(tk.Frame):
    """
    Grid hiển thị thumbnails với scores
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#f0f0f0", **kwargs)
        
        # Canvas for scrolling
        self.canvas = tk.Canvas(self, bg="#f0f0f0", highlightthickness=0)
        self.frame = tk.Frame(self.canvas, bg="#f0f0f0")
        
        # Horizontal scrollbar
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal",
                                        command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set)
        
        self.canvas.pack(side=tk.TOP, fill="both", expand=True, padx=20, pady=(0, 10))
        self.h_scrollbar.pack(side=tk.BOTTOM, fill="x")
        
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        
        # Bind events
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.frame.bind("<Configure>", self._update_scroll_region)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scroll"""
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _update_scroll_region(self, event):
        """Update canvas scroll region"""
        region = self.canvas.bbox("all")
        if region:
            self.canvas.configure(scrollregion=region)
    
    def set_thumbnails(self, paths: List[str], scores: List[float]):
        """
        Hiển thị thumbnails
        
        Args:
            paths: Danh sách đường dẫn ảnh
            scores: Danh sách điểm số (0-1)
        """
        # Clear old widgets
        for widget in self.frame.winfo_children():
            widget.destroy()
        
        # Title
        title = tk.Label(self.frame, text="TOP 4 ẢNH TƯƠNG TỰ (2-5)",
                        font=('Helvetica', 12, 'bold'), fg=PRIMARY, bg=CARD_COLOR)
        title.pack(anchor='w', pady=(5, 8))
        
        # Note
        note = tk.Label(self.frame, text="→ Độ tương tự khớp 100% với bảng kết quả",
                       font=FONT_SMALL, fg=TEXT_LIGHT, bg=CARD_COLOR)
        note.pack(anchor='w', padx=10)
        
        # Row container
        row = tk.Frame(self.frame, bg="#e0e0e0", relief='raised', bd=1)
        row.pack(fill='x', padx=10, pady=5)
        
        # Add thumbnails
        for path, sim in zip(paths, scores):
            try:
                img = Image.open(path).convert("RGB").resize(
                    THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(img)
                
                # Column frame
                col = tk.Frame(row, bg="#d0d0d0", width=110, height=130,
                             highlightbackground=BORDER, highlightthickness=2)
                col.pack(side='left', padx=10, pady=5)
                col.pack_propagate(False)
                
                # Image
                lbl_img = tk.Label(col, image=imgtk, bg=CARD_COLOR,
                                 relief='solid', bd=1)
                lbl_img.image = imgtk  # Keep reference
                lbl_img.pack(pady=(8, 3))
                
                # Score
                score_pct = sim * 100
                color = SUCCESS if score_pct >= 90 else ("orange" if score_pct >= 80 else TEXT_DARK)
                tk.Label(col, text=f"{score_pct:.1f}%",
                        font=('Consolas', 10, 'bold'), fg=color,
                        bg=CARD_COLOR).pack()
                
            except Exception as e:
                logger.error(f"Lỗi hiển thị thumbnail: {e}")
        
        # Update scroll region
        self.winfo_toplevel().after(100, self._update_scroll_region, None)
    
    def clear(self):
        """Xóa tất cả thumbnails"""
        for widget in self.frame.winfo_children():
            widget.destroy()


class InfoLabel(tk.Label):
    """Label hiển thị thông tin với style đẹp"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, font=('Helvetica', 12, 'bold'),
                        fg=PRIMARY, bg=CARD_COLOR, **kwargs)
    
    def set_text(self, text: str, color: str = None):
        """
        Set text và màu
        
        Args:
            text: Nội dung
            color: Màu chữ (optional)
        """
        self.config(text=text)
        if color:
            self.config(fg=color)
