# =============================================================================
# merge_datasets.py
# Merges all extracted YOLOv5 datasets into one unified dataset.
#
# Prerequisites:
#   - Run extract_datasets.py first  (unzip)
#   - Run convert_datasets.py first  (polygon → bbox on SafeWalkBD)
#   - Run remap_classes.py first     (N raw classes → 10 unified classes)
#
# Parallelism:
#   File copying is parallelized with ThreadPoolExecutor. shutil.copy2 is
#   I/O-bound and releases the GIL, so threads give real throughput gains
#   when copying thousands of images across splits and datasets.
#   ProcessPoolExecutor is NOT used here — the overhead of spawning processes
#   for simple file copies would outweigh the benefit.
#
# Pipeline:
#   1. Merge all splits (train/valid/test) into MERGED_DIR in parallel
#      — empty label files (all classes dropped) are skipped with their image
#   2. Write the unified data.yaml from const.py
#
# Usage:
#   python scripts/merge_datasets.py
#   python scripts/merge_datasets.py --workers 8
# =============================================================================

import argparse
import os
import shutil
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from const import (
    EXTRACTED_DIR,
    MERGED_DIR,
    NUM_CLASSES,
    SPLITS,
    UNIFIED_CLASSES,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_WORKERS  = min(8, (os.cpu_count() or 4))


# =============================================================================
# Helpers
# =============================================================================

def find_image(images_dir: Path, stem: str) -> Path | None:
    """Return the image file matching a label stem, or None."""
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def label_is_empty(label_path: Path) -> bool:
    """Return True if a label file has no annotation lines."""
    return not label_path.read_text().strip()


# =============================================================================
# Copy worker
# =============================================================================

def copy_pair(
    label_path: Path,
    images_dir: Path,
    dst_labels_dir: Path,
    dst_images_dir: Path,
    prefix: str,
) -> str:
    """
    Copy one label + its paired image into the merged directories.
    Returns 'copied', 'empty' (skipped), or 'no_image' (label copied, image missing).
    Designed to run inside a thread — no shared mutable state.
    """
    if label_is_empty(label_path):
        return "empty"

    # Copy label
    dst_label = dst_labels_dir / f"{prefix}__{label_path.name}"
    shutil.copy2(label_path, dst_label)

    # Copy paired image
    img = find_image(images_dir, label_path.stem)
    if img:
        dst_img = dst_images_dir / f"{prefix}__{img.name}"
        shutil.copy2(img, dst_img)
        return "copied"

    return "no_image"


# =============================================================================
# Step 1 — Merge splits
# =============================================================================

def merge_splits(max_workers: int) -> None:
    """
    Walk every extracted dataset and copy images + labels into MERGED_DIR,
    preserving train/valid/test splits. File copies run in a thread pool.
    """
    if not EXTRACTED_DIR.exists():
        print("[error] extracted dir not found — run extract_datasets.py first")
        sys.exit(1)

    # Pre-create all destination directories (not thread-safe to do inside workers)
    for split in SPLITS:
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    total_copied = total_dropped = total_no_image = 0

    dataset_dirs = sorted(d for d in EXTRACTED_DIR.iterdir() if d.is_dir())
    print(f"  Workers: {max_workers}  |  Datasets: {len(dataset_dirs)}\n")

    for dataset_dir in dataset_dirs:
        prefix = dataset_dir.name
        futures = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for split in SPLITS:
                labels_dir = dataset_dir / split / "labels"
                images_dir = dataset_dir / split / "images"
                dst_labels = MERGED_DIR / split / "labels"
                dst_images = MERGED_DIR / split / "images"

                if not labels_dir.exists():
                    print(f"  [warn] missing: {dataset_dir.name}/{split}/labels")
                    continue

                for label_path in sorted(labels_dir.glob("*.txt")):
                    future = executor.submit(
                        copy_pair,
                        label_path,
                        images_dir,
                        dst_labels,
                        dst_images,
                        prefix,
                    )
                    futures[future] = label_path.name

        # Tally results
        ds_copied = ds_dropped = ds_no_image = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                if result == "copied":
                    ds_copied += 1
                elif result == "empty":
                    ds_dropped += 1
                elif result == "no_image":
                    ds_no_image += 1
                    print(f"  [warn] no image for label: {futures[future]}")
            except Exception as exc:
                print(f"  [error] {futures[future]}: {exc}")

        print(
            f"[merge]   {prefix:<55} "
            f"copied {ds_copied:>5} | "
            f"dropped {ds_dropped:>5} empty | "
            f"missing img {ds_no_image:>3}"
        )
        total_copied   += ds_copied
        total_dropped  += ds_dropped
        total_no_image += ds_no_image

    print(
        f"\n  Total copied: {total_copied} | "
        f"dropped (empty): {total_dropped} | "
        f"missing images: {total_no_image}"
    )


# =============================================================================
# Step 2 — Write unified data.yaml
# =============================================================================

def build_yaml() -> None:
    """Write the unified data.yaml to MERGED_DIR from const.py definitions."""
    out = MERGED_DIR / "data.yaml"
    with open(out, "w") as f:
        yaml.dump(
            {
                "train": str(MERGED_DIR / "train" / "images"),
                "val":   str(MERGED_DIR / "valid" / "images"),
                "test":  str(MERGED_DIR / "test"  / "images"),
                "nc":    NUM_CLASSES,
                "names": UNIFIED_CLASSES,
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    print(f"[yaml]    {NUM_CLASSES} classes → {out}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all extracted datasets into one unified dataset."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Parallel file-copy threads (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  cv-vision — Dataset Merge")
    print("=" * 60)

    print("\n── Step 1: Merge splits ──────────────────────────────────")
    merge_splits(max_workers=args.workers)

    print("\n── Step 2: Write unified data.yaml ───────────────────────")
    build_yaml()

    print(f"\n✅  Done. Merged dataset → {MERGED_DIR}")


if __name__ == "__main__":
    main()
