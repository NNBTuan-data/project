import os
import requests
import random
import time
import logging
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from ..config import (DATASET_DIR, CLASSES, NUM_IMAGES_PER_CLASS, 
                      MAX_DOWNLOAD_WORKERS, MIN_IMAGE_SIZE, DOWNLOAD_TIMEOUT)
from ddgs import DDGS
logger = logging.getLogger(__name__)
SEARCH_QUERIES = {
    "chair": ["chair", "chair furniture", "wooden chair", "office chair"],
    "table": ["table", "table furniture", "dining table", "wooden table"],
    "bottle": ["bottle", "water bottle", "glass bottle", "plastic bottle"],
    "cup": ["cup", "coffee cup", "tea cup", "mug"],
    "man": ["man portrait", "man face", "adult male", "man person"],
    "woman": ["woman portrait", "woman face", "adult female", "woman person"],
    "child": ["child", "kid", "young child", "child portrait"],
    "people": ["people group", "group of people", "crowd", "multiple people"]
}
class DatasetDownloader:
    def __init__(self, output_dir: Path = None, max_workers: int = None):
        self.output_dir = output_dir or DATASET_DIR
        self.max_workers = max_workers or MAX_DOWNLOAD_WORKERS
        self.print_lock = Lock()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Khởi tạo DatasetDownloader: output={self.output_dir}, workers={self.max_workers}")
    def download_image(self, args: Tuple[str, str, str, int]) -> bool:
        url, path, cls, idx = args
        try:
            time.sleep(0.2)
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT)
            if response.status_code == 200 and len(response.content) > MIN_IMAGE_SIZE:
                with open(path, "wb") as f:
                    f.write(response.content)
                with self.print_lock:
                    logger.debug(f"[{cls}] {idx:04d}")
                    print(f"  [{cls}] {idx:04d}")
                return True
        except Exception as e:
            logger.debug(f"Lỗi tải {url}: {e}")
        return False
    def download_class(self, class_name: str, num_images: int = NUM_IMAGES_PER_CLASS) -> int:
        #tạo folder
        folder = self.output_dir / class_name
        folder.mkdir(exist_ok=True)
        #xóa file cũ trong folder
        existing_files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        if existing_files:
            logger.info(f"Xóa {len(existing_files)} file cũ trong folder {class_name}...")
            for old_file in existing_files:
                try:
                    os.remove(old_file)
                except Exception as e:
                    logger.warning(f"Không xóa được {old_file}: {e}")
        logger.info(f"Đang tải {num_images} ảnh cho class: {class_name}")
        print(f"\nĐang tải {num_images} ảnh cho: {class_name}")
        start_time = time.time()
        all_results = []
        downloaded = 0
        queries = SEARCH_QUERIES.get(class_name, [class_name])
        logger.info(f"Sử dụng {len(queries)} từ khóa tìm kiếm: {queries}")
        for query in queries:
            try:
                time.sleep(random.uniform(2.0, 4.0))
                logger.info(f"  Đang tìm kiếm: '{query}'...")
                print(f"  Tìm kiếm: '{query}'...")
                with DDGS() as ddgs:
                    results = list(ddgs.images(query, max_results=100))
                if not results or isinstance(results, bool):
                    logger.warning(f"    → Không có kết quả")
                    continue
                new_count = 0
                for r in results:
                    if isinstance(r, dict) and "image" in r:
                        url = r["image"]
                        if url not in [res["image"] for res in all_results]:
                            all_results.append(r)
                            new_count += 1
                logger.info(f"    → Tìm thấy {len(results)} ảnh, thêm được {new_count} unique")
                print(f"    → +{new_count} ảnh mới (tổng: {len(all_results)})")
                if len(all_results) >= num_images * 3:
                    logger.info(f"  Đã có đủ {len(all_results)} URLs, dừng search")
                    break
            except Exception as e:
                logger.error(f"  Lỗi khi search '{query}': {e}")
                print(f"  ✗ Lỗi: {e}")
        if not all_results:
            logger.warning(f"Không có kết quả nào từ DuckDuckGo cho {class_name}")
            print(f"  Không có kết quả nào từ DuckDuckGo cho {class_name}")
            return 0
        random.shuffle(all_results)
        logger.info(f"Tổng cộng: {len(all_results)} URLs unique")
        tasks = []
        max_tasks = min(len(all_results), num_images * 3)  # 3x để đảm bảo
        for i in range(max_tasks):
            r = all_results[i]
            url = r["image"]
            # Sử dụng tên TEMP để tránh xung đột 
            temp_filename = folder / f"temp_{i:05d}.jpg"
            tasks.append((url, str(temp_filename), class_name, i+1))
        logger.info(f"Tạo {len(tasks)} tasks để tải (mục tiêu: {num_images} ảnh)")
        downloaded = 0
        successful_files = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_image, task): task for task in tasks}
            for future in as_completed(futures):
                if future.result():
                    task = futures[future]
                    successful_files.append(task[1])
                    downloaded += 1
                    # Dừng sớm nếu đã đủ số lượng
                    if downloaded >= num_images:
                        logger.info(f"Đã đủ {num_images} ảnh, hủy các tasks còn lại...")
                        # Hủy các tasks còn lại 
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
        logger.info(f"Tải thành công {len(successful_files)} ảnh")
        final_count = 0
        if successful_files:
            kept_count = min(len(successful_files), num_images)
            logger.info(f"Đang rename {kept_count} ảnh thành công...")
            for idx in range(kept_count):
                temp_path = successful_files[idx]
                final_filename = folder / f"{class_name}_{idx+1:04d}.jpg"
                if os.path.exists(temp_path):
                    try:
                        if os.path.exists(str(final_filename)):
                            os.remove(str(final_filename))
                        os.rename(temp_path, str(final_filename))
                        final_count += 1
                    except Exception as e:
                        logger.error(f"Lỗi rename {temp_path} → {final_filename}: {e}")
                else:
                    logger.warning(f"File temp không tồn tại: {temp_path}")
            for idx in range(kept_count, len(successful_files)):
                temp_path = successful_files[idx]
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.warning(f"Không xóa được {temp_path}: {e}")
        elapsed = time.time() - start_time
        logger.info(f"Hoàn thành: {class_name} → {final_count}/{num_images} ảnh (mất {elapsed:.1f}s)")
        print(f"Hoàn thành: {class_name} → {final_count}/{num_images} ảnh (mất {elapsed:.1f}s)\n")
        return final_count
    def download_all(self, classes: List[str] = None, num_images: int = NUM_IMAGES_PER_CLASS) -> dict:
        classes = classes or CLASSES
        logger.info(f"BẮT ĐẦU TẢI {len(classes)} CLASSES")
        print("=" * 70)
        print(f"BẮT ĐẦU TẢI {len(classes)} CLASSES")
        print("=" * 70)
        results = {}
        total_start = time.time()
        for cls in classes:
            downloaded = self.download_class(cls, num_images)
            results[cls] = downloaded
        total_elapsed = time.time() - total_start
        total_images = sum(results.values())
        logger.info(f"Hoàn thành tải: {total_images} ảnh trong {total_elapsed:.1f}s")
        print(f"Đã tải xong {total_images} Ảnh!")
        print(f"Thời gian: {total_elapsed:.1f}s")
        print(f"Kiểm tra thư mục: {self.output_dir}")
        return results
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    downloader = DatasetDownloader()
    test_classes = ["chair"]
    results = downloader.download_all(test_classes, num_images=10)
    
    print(f"\nKết quả: {results}")
