import os
from pathlib import Path
from PIL import Image
from collections import Counter
from tqdm import tqdm

# Update this path to match your project structure
MERGED_DIR = Path("src/cv-vision/data/merged")

def check_resolutions():
    if not MERGED_DIR.exists():
        print(f"[error] Merged directory not found at {MERGED_DIR}")
        return

    resolutions = []
    image_files = list(MERGED_DIR.rglob("images/*.*"))
    
    print(f"Checking {len(image_files)} images...")

    for img_path in tqdm(image_files):
        # We use PIL because it's faster than OpenCV just for reading metadata
        try:
            with Image.open(img_path) as img:
                resolutions.append(img.size) # returns (width, height)
        except Exception as e:
            print(f"[warn] Could not read {img_path}: {e}")

    # Count occurrences of each resolution
    stats = Counter(resolutions)

    print("\n" + "="*30)
    print("  IMAGE RESOLUTION REPORT")
    print("="*30)
    
    # Sort by frequency
    for (w, h), count in stats.most_common():
        percentage = (count / len(image_files)) * 100
        print(f"  {w}x{h} : {count} images ({percentage:.2f}%)")

if __name__ == "__main__":
    check_resolutions()
