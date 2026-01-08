import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.data.dataset_downloader import DatasetDownloader
import logging

logging.basicConfig(level=logging.INFO)

def main():
    downloader = DatasetDownloader()
    print("Testing download for 'chair'...")
    # Try downloading just 1 image for 'chair'
    # Since download_all or download_class usually does more, 
    # I'll manually call download_class but with num_images=1
    
    # Need to override SEARCH_QUERIES temporarily to search just "chair" once
    # but the logic handles 1 image fine.
    
    count = downloader.download_class("chair", num_images=1)
    print(f"Downloaded {count} images.")

if __name__ == "__main__":
    main()
