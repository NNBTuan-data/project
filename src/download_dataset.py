
import os
import requests
import random
import time
from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
classes = [
    "chair", "table", "bottle", "cup",
    "man", "woman", "child", "people"
]

num_images = 100   # số ảnh mỗi lớp
output_dir = "dataset"
max_workers = 10   # số luồng tải song song

os.makedirs(output_dir, exist_ok=True)
print_lock = Lock()

def download_image(args):
    url, path, cls, idx = args
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200 and len(response.content) > 8000:
            with open(path, "wb") as f:
                f.write(response.content)
            with print_lock:
                print(f"  [{cls}] {idx:04d} ✓")
            return True
    except:
        pass
    return False

print("BẮT ĐẦU TẢI CÁC LỚP BỊ THIẾU...\n")

with DDGS() as ddgs:
    for cls in classes:
        folder = os.path.join(output_dir, cls)
        os.makedirs(folder, exist_ok=True)
        print(f"Đang tải {num_images} ảnh cho: {cls} ...")

        start_time = time.time()
        count = 0

        try:
            results = list(ddgs.images(cls, max_results=300))
            if not results or isinstance(results, bool):
                print(f"  Không có kết quả cho {cls}")
                continue
            random.shuffle(results)
        except Exception as e:
            print(f"  Lỗi API: {e}")
            continue

        tasks = []
        for i, r in enumerate(results):
            if count >= num_images:
                break
            if isinstance(r, dict) and "image" in r:
                url = r["image"]
                filename = os.path.join(folder, f"{cls}_{count+1:04d}.jpg")
                tasks.append((url, filename, cls, count+1))
                count += 1

        downloaded = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_image, task) for task in tasks]
            for future in as_completed(futures):
                if future.result():
                    downloaded += 1

        elapsed = time.time() - start_time
        print(f"HOÀN THÀNH: {cls} → {downloaded} ảnh (mất {elapsed:.1f}s)\n")

print("✅ ĐÃ TẢI BỔ SUNG XONG 8 LỚP BỊ THIẾU!")
print("Kiểm tra thư mục dataset/ để xem kết quả.")
