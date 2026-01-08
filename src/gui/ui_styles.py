"""
UI Styles và Constants
Theme: Modern Dark với Gradient
"""

# ================================
# MÀU SẮC - DARK THEME
# ================================
BG_COLOR = "#0f172a"       # Dark blue
CARD_COLOR = "#1e293b"     # Slate
PRIMARY = "#3b82f6"        # Bright blue
SECONDARY = "#8b5cf6"      # Purple
SUCCESS = "#10b981"        # Green
DANGER = "#ef4444"         # Red
ACCENT = "#06b6d4"         # Cyan
TEXT_DARK = "#f1f5f9"      # Light text
TEXT_LIGHT = "#94a3b8"     # Muted text
BORDER = "#334155"         # Border

# ================================
# KÍCH THƯỚC
# ================================
# Cửa sổ chính - TO HƠN
WINDOW_SIZE = "1600x900"

# Frame sizes - TO HƠN
PREVIEW_FRAME_SIZE = (520, 520)
THUMBNAIL_SIZE = (120, 120)
PREVIEW_IMAGE_SIZE = (500, 500)

# ================================
# FONTS - HIỆN ĐẠI HƠN
# ================================
FONT_TITLE = ('Segoe UI', 24, 'bold')
FONT_HEADER = ('Segoe UI', 14, 'bold')
FONT_BUTTON = ('Segoe UI', 13, 'bold')
FONT_TEXT = ('Consolas', 11)
FONT_RESULT = ('Consolas', 16, 'bold')
FONT_ITALIC = ('Segoe UI', 11, 'italic')
FONT_SMALL = ('Segoe UI', 9)

# ================================
# PADDING & SPACING
# ================================
BUTTON_PADDING = 15
FRAME_PADDING = 25
CARD_PADDING = 20

# ================================
# TTK STYLE CONFIGURATION
# ================================
def configure_ttk_styles(style):
    """
    Cấu hình ttk styles với dark theme
    
    Args:
        style: ttk.Style object
    """
    style.theme_use('clam')
    
    # Label styles
    style.configure('Title.TLabel', 
                   font=FONT_TITLE, 
                   foreground=PRIMARY, 
                   background=BG_COLOR)
    
    style.configure('Header.TLabel', 
                   font=FONT_HEADER, 
                   foreground=TEXT_DARK, 
                   background=CARD_COLOR)
    
    # Button styles - Modern với hover effect
    style.configure('TButton', 
                   font=FONT_BUTTON, 
                   padding=BUTTON_PADDING,
                   background=PRIMARY,
                   foreground='white',
                   borderwidth=0)
    
    style.map('TButton', 
             background=[('active', SECONDARY), ('pressed', ACCENT)], 
             foreground=[('active', 'white')])


# ================================
# TEXT TAG CONFIGURATIONS
# ================================
TEXT_TAGS = {
    "header": {
        "foreground": PRIMARY,
        "font": ("Consolas", 14, "bold")
    },
    "header_svm": {
        "foreground": SUCCESS,
        "font": ("Consolas", 13, "bold")
    },
    "final": {
        "foreground": ACCENT,
        "font": ("Consolas", 15, "bold")
    },
    "note": {
        "foreground": TEXT_LIGHT,
        "font": ("Consolas", 10, "italic")
    },
    "line": {
        "foreground": BORDER
    },
    "error": {
        "foreground": DANGER
    }
}
