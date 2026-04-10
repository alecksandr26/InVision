import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image  # Faster for metadata-only checks

# Import project constants
from const import MERGED_DIR, SPLITS, TARGET_SIZE, MIN_DIM


def letterbox_image(img, target_size=640):
    """Resizes and pads image to square without stretching."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # INTER_AREA is best for shrinking; INTER_LINEAR is faster
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def process_single_file(img_path: Path):
    """Worker: Filter and Resize image, delete if too small."""
    try:
        # Optimization 1: Fast metadata check using PIL
        # This prevents loading a 4K image into RAM if it's already 640x640
        with Image.open(img_path) as meta:
            width, height = meta.size
        
        # 1. FILTER: Check if image is too small
        if width < MIN_DIM or height < MIN_DIM:
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            img_path.unlink(missing_ok=True)
            if label_path.exists():
                label_path.unlink()
            return "deleted"

        # 2. SKIP: If already correct size, don't even open with OpenCV
        if width == TARGET_SIZE and height == TARGET_SIZE:
            return "skipped"

        # 3. RESIZE: Only now do we load the full pixels
        img = cv2.imread(str(img_path))
        if img is None:
            return "error"

        cleaned_img = letterbox_image(img, TARGET_SIZE)
        cv2.imwrite(str(img_path), cleaned_img)
        return "resized"

    except Exception:
        return "error"

def main():
    parser = argparse.ArgumentParser(description="Clean and Resize InVision Dataset")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("  cv-vision — Dataset Cleaner & Resizer")
    print("=" * 60)

    # Use the absolute path to be 100% sure where we are looking
    target_path = MERGED_DIR.resolve()
    print(f"Target Directory: {target_path}")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = []
    
    # We iterate through the standard YOLO splits defined in const.py
    for split in SPLITS:
        # In YOLOv5/v8, images are usually in merged/<split>/images/
        img_dir = target_path / split / "images"
        
        if img_dir.exists():
            # Use glob to ensure we catch all variations of extensions
            found = [p for p in img_dir.glob("*") if p.suffix.lower() in valid_exts]
            print(f"  [+] Found {len(found):5} images in '{split}'")
            image_files.extend(found)
        else:
            print(f"  [!] Directory not found: {img_dir}")

    if not image_files:
        print("\n[error] No images found. Check if your merged folder structure is:")
        print(f"        {target_path}/[train|valid|test]/images/")
        return

    print(f"\nStarting parallel processing of {len(image_files)} images...")
    
    stats = {"deleted": 0, "resized": 0, "skipped": 0, "error": 0}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, img): img for img in image_files}
        
        # tqdm will now show the actual progress bar for 24,982 files
        for future in tqdm(as_completed(futures), total=len(image_files), desc="Cleaning"):
            result = future.result()
            stats[result] += 1

    print("\n" + "─" * 40)
    print(f"  CLEANUP COMPLETE")
    print(f"  Discarded (Too small): {stats['deleted']}")
    print(f"  Resized/Letterboxed:  {stats['resized']}")
    print(f"  Already 640x640:      {stats['skipped']}")
    print(f"  Errors:               {stats['error']}")
    print("─" * 40)

if __name__ == "__main__":
    main()
