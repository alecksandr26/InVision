import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

# Import project constants
from const import MERGED_DIR, SPLITS, TARGET_SIZE, MIN_DIM

def letterbox_image_and_labels(img, label_lines, target_size=640):
    """
    Resizes image to square, adds padding, AND scales YOLO labels.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 1. Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 2. Create blank canvas and paste resized image
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # 3. Adjust Bounding Boxes
    new_labels = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) < 5: continue
        
        cls_id = parts[0]
        x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts[1:5])
        
        # Convert normalized (0-1) of ORIGINAL image to absolute pixels
        abs_x_center = x_center_norm * w
        abs_y_center = y_center_norm * h
        abs_w = w_norm * w
        abs_h = h_norm * h
        
        # Scale to new inner image size
        scaled_x_center = abs_x_center * scale
        scaled_y_center = abs_y_center * scale
        scaled_w = abs_w * scale
        scaled_h = abs_h * scale
        
        # Add the padding offsets
        final_x_center = scaled_x_center + x_offset
        final_y_center = scaled_y_center + y_offset
        
        # Convert back to normalized (0-1) of the NEW SQUARE TARGET canvas
        new_x_norm = final_x_center / target_size
        new_y_norm = final_y_center / target_size
        new_w_norm = scaled_w / target_size
        new_h_norm = scaled_h / target_size
        
        # Format back to YOLO string
        new_line = f"{cls_id} {new_x_norm:.6f} {new_y_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f}"
        new_labels.append(new_line)
        
    return canvas, new_labels

def process_single_file(img_path: Path):
    """Worker: Filter, Resize image, UPDATE labels, delete if too small."""
    try:
        # Fast metadata check
        with Image.open(img_path) as img_pil:
            w, h = img_pil.size
            if min(w, h) < MIN_DIM:
                # Need to delete both image and label
                label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
                img_path.unlink()
                if label_path.exists():
                    label_path.unlink()
                return "deleted"
            
            # If already correct size, skip
            if w == TARGET_SIZE and h == TARGET_SIZE:
                return "skipped"

        # If we need to resize, read via OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            return "error"

        # Find corresponding label
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        label_lines = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                label_lines = f.readlines()

        # Perform letterbox AND label shift
        new_img, new_labels = letterbox_image_and_labels(img, label_lines, TARGET_SIZE)

        # Overwrite the image
        cv2.imwrite(str(img_path), new_img)

        # Overwrite the labels with the new coordinates
        if new_labels:
            with open(label_path, 'w') as f:
                f.write("\n".join(new_labels) + "\n")

        return "resized"

    except Exception as e:
        print(f"\n[error] Failed on {img_path.name}: {e}")
        return "error"

def main():
    parser = argparse.ArgumentParser(description="Clean, resize images, and update YOLO labels.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel processing threads")
    args = parser.parse_args()

    target_path = MERGED_DIR
    image_files = []
    valid_exts = {'.jpg', '.jpeg', '.png'}

    for split in SPLITS:
        img_dir = target_path / split / "images"
        if img_dir.exists():
            found = [p for p in img_dir.glob("*") if p.suffix.lower() in valid_exts]
            image_files.extend(found)

    if not image_files:
        print("[error] No images found.")
        return

    print(f"Starting pipeline on {len(image_files)} images...")
    stats = {"deleted": 0, "resized": 0, "skipped": 0, "error": 0}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, img): img for img in image_files}
        
        for future in tqdm(as_completed(futures), total=len(image_files), desc="Cleaning & Re-labeling"):
            result = future.result()
            stats[result] += 1

    print("\n" + "─" * 40)
    print(f"  PIPELINE COMPLETE")
    print(f"  Resized & Relabeled: {stats['resized']}")
    print(f"  Skipped (Already 640): {stats['skipped']}")
    print(f"  Deleted (Too Small): {stats['deleted']}")
    print(f"  Errors: {stats['error']}")
    print("─" * 40)

if __name__ == "__main__":
    main()
