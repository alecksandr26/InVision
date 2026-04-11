import cv2
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Import your central configuration
from const import MERGED_DIR, UNIFIED_CLASSES, SPLITS

def find_and_preview(target_filename):
    """
    Search for an image in the merged dataset splits and display its labels.
    """
    image_path = None
    label_path = None
    found_split = None

    # 1. Search through splits (train, valid, test) defined in const.py
    for split in SPLITS:
        potential_img = MERGED_DIR / split / "images" / target_filename
        if potential_img.exists():
            image_path = potential_img
            # Labels usually have the same name but .txt extension
            label_filename = image_path.stem + ".txt"
            label_path = MERGED_DIR / split / "labels" / label_filename
            found_split = split
            break

    if not image_path:
        print(f"❌ Error: Could not find '{target_filename}' in {MERGED_DIR}")
        return

    print(f"✅ Found in: {found_split}")
    print(f"🖼️  Image: {image_path}")
    print(f"📝 Label: {label_path}")

    # 2. Load the image
    img = cv2.imread(str(image_path))
    if img is None:
        print("❌ Error: Failed to load image data.")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # 3. Draw Labels if they exist
    if label_path and label_path.exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            cls_id = int(parts[0])
            x_c, y_c, bw, bh = map(float, parts[1:])


            # Convert YOLO normalized (0-1) to Pixel Coordinates
            x1 = int((x_c - bw/2) * w)
            y1 = int((y_c - bh/2) * h)
            x2 = int((x_c + bw/2) * w)
            y2 = int((y_c + bh/2) * h)

            # Get class name from your UNIFIED_CLASSES list
            class_name = UNIFIED_CLASSES[cls_id] if cls_id < len(UNIFIED_CLASSES) else "Unknown"
            print(f"class name: {class_name} - bbox: {[x_c, y_c, bw, bh]}")

            # Draw Box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label Background & Text
            label_str = f"{class_name} ({cls_id})"
            cv2.putText(img, label_str, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print("⚠️  Warning: No label file found for this image.")

    # 4. Show using Matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(f"Preview: {target_filename} ({found_split})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run via command line: python preview_image.py my_image.jpg
        find_and_preview(sys.argv[1])
    else:
        print("💡 Usage: python preview_image.py <image_name.jpg>")
        # Example hardcoded test if you just want to run it:
        # find_and_preview("example_01.jpg")

