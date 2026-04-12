# =============================================================================
# build.py
# Packages the merged dataset into a distributable zip archive.
#
# Prerequisites:
#   - Run extract_datasets.py first  (unzip)
#   - Run convert_datasets.py first  (polygon → bbox on SafeWalkBD)
#   - Run remap_classes.py first     (N raw classes → 10 unified classes)
#   - Run merge_datasets.py first    (merge splits + write data.yaml)
#
# Output:
#   data/invision-dataset-v<VERSION>.zip
#   └── invision-dataset/
#       ├── data.yaml
#       ├── train/
#       │   ├── images/
#       │   └── labels/
#       ├── valid/
#       │   ├── images/
#       │   └── labels/
#       └── test/
#           ├── images/
#           └── labels/
#
# Usage:
#   python scripts/build.py
#   python scripts/build.py --version 2
# =============================================================================

import argparse
import sys
import zipfile
from pathlib import Path

from const import DATA_DIR, MERGED_DIR, NUM_CLASSES, SPLITS, UNIFIED_CLASSES

# =============================================================================
# Config
# =============================================================================

DATASET_NAME    = "invision-dataset"
DEFAULT_VERSION = 1


# =============================================================================
# Validation
# =============================================================================

def validate_merged() -> bool:
    """
    Check that the merged dataset is complete before packaging.
    Verifies data.yaml and that every split has at least one image and label.
    Returns True if valid, False otherwise.
    """
    ok = True

    yaml_path = MERGED_DIR / "data.yaml"
    if not yaml_path.exists():
        print(f"  [error] missing data.yaml in {MERGED_DIR}")
        ok = False

    for split in SPLITS:
        images_dir = MERGED_DIR / split / "images"
        labels_dir = MERGED_DIR / split / "labels"

        for d in (images_dir, labels_dir):
            if not d.exists():
                print(f"  [error] missing directory: {d}")
                ok = False
                continue

            files = list(d.iterdir())
            if not files:
                print(f"  [warn] empty directory: {d}")

    return ok


# =============================================================================
# Stats
# =============================================================================

def print_stats() -> None:
    """Print a summary of the merged dataset before packaging."""
    print(f"\n  Classes ({NUM_CLASSES}):")
    for i, cls in enumerate(UNIFIED_CLASSES):
        print(f"    [{i}] {cls}")

    print(f"\n  Split summary:")
    for split in SPLITS:
        images = list((MERGED_DIR / split / "images").iterdir())
        labels = list((MERGED_DIR / split / "labels").iterdir())
        print(f"    {split:<6}  {len(images):>6} images  {len(labels):>6} labels")


# =============================================================================
# Build
# =============================================================================

def build_zip(version: int) -> Path:
    """
    Package the merged dataset into a zip archive under DATA_DIR.

    Archive internal structure:
        invision-dataset/
            data.yaml
            train/images/  train/labels/
            valid/images/  valid/labels/
            test/images/   test/labels/

    Returns the path to the created zip file.
    """
    zip_name = f"{DATASET_NAME}-v{version}.zip"
    zip_path = DATA_DIR / zip_name

    if zip_path.exists():
        print(f"  [warn] overwriting existing archive: {zip_name}")
        zip_path.unlink()

    total_files = 0

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        # data.yaml
        yaml_path = MERGED_DIR / "data.yaml"
        zf.write(yaml_path, arcname=f"{DATASET_NAME}/data.yaml")
        total_files += 1

        # splits
        for split in SPLITS:
            for kind in ("images", "labels"):
                src_dir = MERGED_DIR / split / kind
                if not src_dir.exists():
                    continue

                for file in sorted(src_dir.iterdir()):
                    arcname = f"{DATASET_NAME}/{split}/{kind}/{file.name}"
                    zf.write(file, arcname=arcname)
                    total_files += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n  Archived {total_files} files → {zip_name}  ({size_mb:.1f} MB)")

    return zip_path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Package the merged dataset into {DATASET_NAME}-v<N>.zip"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=DEFAULT_VERSION,
        metavar="N",
        help=f"Dataset version number (default: {DEFAULT_VERSION})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  cv-vision — Build Dataset Archive")
    print("=" * 60)

    if not MERGED_DIR.exists():
        print("[error] merged directory not found — run merge_datasets.py first")
        sys.exit(1)

    print("\n── Validating merged dataset ─────────────────────────────")
    if not validate_merged():
        print("\n[error] validation failed — fix the issues above and retry")
        sys.exit(1)

    print_stats()

    print("\n── Packaging ─────────────────────────────────────────────")
    zip_path = build_zip(version=args.version)

    print(f"\n✅  Done. Archive → {zip_path}")


if __name__ == "__main__":
    main()


