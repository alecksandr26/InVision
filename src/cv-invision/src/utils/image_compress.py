import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from .const import TEST_SAMPLES_DIR, INPUT_IMAGE_SIZE, IMAGE_QUALITY, IMAGE_EXTENSIONS

def compress_single_image(img_path, target_size, quality):
    """Resizes and compresses a single image file."""
    try:
        with Image.open(img_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            size = (target_size, target_size)
            # LANCZOS is high-quality for downsampling
            img_resized = img.resize(size, Image.Resampling.LANCZOS)
            
            # Save and overwrite
            img_resized.save(img_path, "JPEG", optimize=True, quality=quality)
            return True
    except Exception as e:
        print(f"\nError processing {img_path.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="InVision Image Compression Tool")
    
    # Arguments
    parser.add_argument("input", type=str, help="Path to a specific image file or a directory")
    parser.add_argument("--size", type=int, default=INPUT_IMAGE_SIZE, help=f"Target square size (default: {INPUT_IMAGE_SIZE})")
    parser.add_argument("--quality", type=int, default=IMAGE_QUALITY, help=f"JPEG quality 1-100 (default: {IMAGE_QUALITY})")
    
    args = parser.parse_args()
    input_path = Path(args.input)

    # Logic to find files
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files = [input_path]
        else:
            print(f"File extension {input_path.suffix} is not in supported list.")
            return
    elif input_path.is_dir():
        image_files = [
            f for f in input_path.iterdir() 
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        print(f"Path not found: {input_path}")
        return

    if not image_files:
        print(f"No valid images found to process.")
        return

    print(f"🚀 Processing {len(image_files)} image(s) at {args.size}x{args.size} (Quality: {args.quality})...")

    # Use a lambda or partial to pass the arguments to the thread executor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(
            executor.map(lambda p: compress_single_image(p, args.size, args.quality), image_files), 
            total=len(image_files)
        ))

    print("\n✅ Compression task finished.")

if __name__ == "__main__":
    main()
