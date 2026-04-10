# =============================================================================
# remap_classes.py
# Remaps all dataset class labels into the 10 unified InVision classes.
#
# All class definitions, mappings, and drop rules live in const.py.
# This script only handles the file I/O and remapping logic.
#
# Prerequisites:
#   - Run extract_datasets.py first  (unzip)
#   - Run convert_datasets.py first  (polygon → bbox on SafeWalkBD)
#
# Usage:
#   python scripts/remap_classes.py
#   python scripts/remap_classes.py --dry-run
# =============================================================================

import argparse
import sys
from pathlib import Path

import yaml

from const import (
    ALL_DATASET_ZIPS,
    CLASS_REMAP,
    EXTRACTED_DIR,
    NUM_CLASSES,
    SPLITS,
    UNIFIED_CLASS_ID,
    UNIFIED_CLASSES,
)


# =============================================================================
# Per-dataset remapper
# =============================================================================

def load_dataset_classes(dataset_dir: Path) -> list[str]:
    """Read class names from a dataset's data.yaml."""
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"  [warn] no data.yaml in {dataset_dir.name}")
        return []
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    return [str(n) for n in meta.get("names", [])]


def remap_label_file(
    label_path: Path,
    old_classes: list[str],
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Remap a single YOLOv5 label file in-place using CLASS_REMAP from const.py.

    Each line: <class_id> <x_center> <y_center> <width> <height>

    - Lines whose raw class maps to a unified class → rewritten with new id
    - Lines whose raw class is not in CLASS_REMAP → dropped (in DROPPED_CLASSES)
    - If all lines are dropped the file is emptied (merge_datasets.py will skip it)

    Returns:
        (kept, dropped) — annotation line counts
    """
    lines = label_path.read_text().splitlines()
    new_lines = []
    kept = dropped = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        try:
            old_id = int(parts[0])
            coords = parts[1:]
        except (ValueError, IndexError):
            dropped += 1
            continue

        if old_id >= len(old_classes):
            dropped += 1
            continue

        raw_name = old_classes[old_id].strip().lower()
        unified_name = CLASS_REMAP.get(raw_name)  # None → dropped

        if unified_name is None:
            dropped += 1
            continue

        new_id = UNIFIED_CLASS_ID[unified_name]
        new_lines.append(f"{new_id} {' '.join(coords)}")
        kept += 1

    if not dry_run:
        label_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))

    return kept, dropped


def remap_dataset(dataset_dir: Path, dry_run: bool = False) -> None:
    """Remap all label files across all splits for one dataset."""
    old_classes = load_dataset_classes(dataset_dir)
    if not old_classes:
        return

    total_kept = total_dropped = images_dropped = 0

    for split in SPLITS:
        labels_dir = dataset_dir / split / "labels"
        images_dir = dataset_dir / split / "images"

        if not labels_dir.exists():
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            kept, dropped = remap_label_file(label_path, old_classes, dry_run)
            total_kept += kept
            total_dropped += dropped

            # No valid labels left → drop image + label from extracted dir.
            # merge_datasets.py will also skip empty label files as a safety net.
            if kept == 0:
                images_dropped += 1
                for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                    img = images_dir / (label_path.stem + ext)
                    if img.exists():
                        if not dry_run:
                            img.unlink()
                        break
                if not dry_run:
                    label_path.unlink()

    tag = "[dry-run] " if dry_run else ""
    print(
        f"  {tag}kept {total_kept} annotations | "
        f"dropped {total_dropped} | "
        f"removed {images_dropped} empty images"
    )


def write_unified_yaml(dataset_dir: Path, dry_run: bool = False) -> None:
    """Overwrite the dataset's data.yaml with the unified class list from const.py."""
    yaml_path = dataset_dir / "data.yaml"
    if dry_run:
        print(f"  [dry-run] would rewrite {yaml_path}")
        return

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    meta["nc"]    = NUM_CLASSES
    meta["names"] = UNIFIED_CLASSES

    with open(yaml_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap dataset classes to the 10 unified InVision classes."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing any files",
    )
    args = parser.parse_args()

    if not EXTRACTED_DIR.exists():
        print("[error] extracted directory not found — run extract_datasets.py first")
        sys.exit(1)

    print("=" * 60)
    print("  cv-vision — Remap Classes")
    print("=" * 60)
    print(f"\nUnified classes ({NUM_CLASSES}):")
    for i, cls in enumerate(UNIFIED_CLASSES):
        print(f"  [{i}] {cls}")
    print()

    for zip_path in ALL_DATASET_ZIPS:
        dataset_dir = EXTRACTED_DIR / zip_path.stem

        if not dataset_dir.exists():
            print(f"[error] not extracted yet: {dataset_dir.name}")
            print("        run extract_datasets.py first")
            sys.exit(1)

        print(f"[remap]   {dataset_dir.name}")
        remap_dataset(dataset_dir, dry_run=args.dry_run)
        write_unified_yaml(dataset_dir, dry_run=args.dry_run)

    print("\n✅  Done.")


if __name__ == "__main__":
    main()
