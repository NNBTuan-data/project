
import sys
from pathlib import Path
from PIL import Image
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATASET_DIR

def check_images():
    print(f"Checking images in {DATASET_DIR}...")
    corrupted_count = 0
    checked_count = 0
    
    # Walk through all directories
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = Path(root) / file
                checked_count += 1
                try:
                    with Image.open(file_path) as img:
                        img.verify() # Verify file integrity
                except Exception as e:
                    print(f"Corrupted found: {file_path} - {e}")
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                        corrupted_count += 1
                    except Exception as del_e:
                        print(f"Could not delete: {del_e}")
                        
    print("\n" + "="*50)
    print(f"SUMMARY")
    print("="*50)
    print(f"Checked: {checked_count} images")
    print(f"Corrupted/Deleted: {corrupted_count} images")
    print("="*50)

if __name__ == "__main__":
    check_images()
