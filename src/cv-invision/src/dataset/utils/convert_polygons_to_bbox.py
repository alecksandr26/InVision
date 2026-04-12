# =============================================================================
# convert_polygons_to_bbox.py
# Converts YOLOv5 polygon label files to axis-aligned bounding box format.
#
# Algorithm:
#   YOLOv5 polygon labels store vertices as normalized (x, y) pairs:
#       <class> x1 y1 x2 y2 x3 y3 ... xN yN
#
#   To find the tightest axis-aligned bbox that contains all polygon points:
#       x_min = min(x1, x2, ..., xN)    ← leftmost point
#       y_min = min(y1, y2, ..., yN)    ← topmost point
#       x_max = max(x1, x2, ..., xN)    ← rightmost point
#       y_max = max(y1, y2, ..., yN)    ← bottommost point
#
#   Then convert to YOLOv5 bbox format (center + dimensions, normalized):
#       x_center = (x_min + x_max) / 2
#       y_center = (y_min + y_max) / 2
#       width    = x_max - x_min
#       height   = y_max - y_min
#
#   Output line per object:
#       <class> x_center y_center width height
#
# Usage:
#   python scripts/convert_polygons_to_bbox.py --dataset-dir data/extracted/<name>
#   python scripts/convert_polygons_to_bbox.py --dataset-dir data/extracted/<name> --dry-run
# =============================================================================

import argparse
from pathlib import Path


# =============================================================================
# Core Algorithm
# =============================================================================

def polygon_to_bbox(values: list[float]) -> tuple[float, float, float, float]:
    """
    Convert a flat list of normalized polygon vertices [x1,y1,x2,y2,...,xN,yN]
    into a YOLOv5 bounding box (x_center, y_center, width, height).

    Strategy: axis-aligned bounding rectangle from min/max extent of all vertices.
    Since all values are already normalized [0, 1], no scaling is needed.

    Args:
        values: flat list of floats [x1, y1, x2, y2, ..., xN, yN]
                must have an even number of elements (at least 4 = 2 points)

    Returns:
        (x_center, y_center, width, height) — all normalized [0, 1]

    Raises:
        ValueError: if fewer than 4 values (need at least 2 vertices to form a box)
    """
    if len(values) < 4 or len(values) % 2 != 0:
        raise ValueError(
            f"Expected an even number of values ≥ 4, got {len(values)}: {values}"
        )

    # Separate x and y coordinates from the interleaved list
    xs = values[0::2]  # indices 0, 2, 4, ... → x1, x2, x3, ...
    ys = values[1::2]  # indices 1, 3, 5, ... → y1, y2, y3, ...

    # Axis-aligned bounding rectangle: corners from min/max extent
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Convert corner form → YOLOv5 center form
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width    = x_max - x_min
    height   = y_max - y_min

    return x_center, y_center, width, height


def convert_label_file(label_path: Path, dry_run: bool = False) -> int:
    """
    Convert a single YOLOv5 label file from polygon to bbox format, in-place.

    Each line format:
        polygon:  <class> x1 y1 x2 y2 ... xN yN   (6+ values after class)
        bbox:     <class> x_center y_center w h     (4 values after class)

    Lines that already have exactly 4 values after the class id (already bbox)
    are left untouched. Lines with fewer than 4 are skipped with a warning.

    Args:
        label_path: path to the .txt label file
        dry_run: if True, print what would change but don't write

    Returns:
        number of lines converted (0 if file was already all bboxes)
    """
    lines = label_path.read_text().splitlines()
    converted_lines = []
    num_converted = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            converted_lines.append("")
            continue

        parts = line.split()
        class_id = parts[0]

        try:
            coords = list(map(float, parts[1:]))
        except ValueError:
            print(f"  [warn] unparseable line {i+1} in {label_path.name}: {line!r}")
            converted_lines.append(line)
            continue

        num_coords = len(coords)

        if num_coords == 4:
            # Already a valid bbox — pass through unchanged
            converted_lines.append(line)

        elif num_coords < 4:
            print(
                f"  [warn] line {i+1} in {label_path.name} has only {num_coords} "
                f"coordinate(s) after class — skipping: {line!r}"
            )
            converted_lines.append(line)

        else:
            # Polygon → bbox
            x_c, y_c, w, h = polygon_to_bbox(coords)
            new_line = f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
            converted_lines.append(new_line)
            num_converted += 1

    if num_converted > 0 and not dry_run:
        label_path.write_text("\n".join(converted_lines) + "\n")

    return num_converted


# =============================================================================
# Dataset Walker
# =============================================================================

def convert_dataset(dataset_dir: Path, dry_run: bool = False) -> None:
    """
    Walk all label files across all splits in a YOLOv5 dataset directory
    and convert any polygon annotations to bboxes in-place.

    Expected structure:
        <dataset_dir>/
            train/labels/*.txt
            valid/labels/*.txt
            test/labels/*.txt

    Args:
        dataset_dir: root of the extracted YOLOv5 dataset
        dry_run: report changes without writing files
    """
    splits = ["train", "valid", "test"]
    total_files = 0
    total_converted_lines = 0

    for split in splits:
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            continue

        label_files = sorted(labels_dir.glob("*.txt"))
        print(f"  [{split}] {len(label_files)} label files")

        for label_path in label_files:
            n = convert_label_file(label_path, dry_run=dry_run)
            if n > 0:
                total_files += 1
                total_converted_lines += n

    tag = "[dry-run] would convert" if dry_run else "converted"
    print(f"  → {tag} {total_converted_lines} polygon annotation(s) across {total_files} file(s)")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLOv5 polygon labels to axis-aligned bounding boxes."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root directory of the extracted YOLOv5 dataset (contains train/valid/test)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be converted without writing any files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset_dir.exists():
        print(f"[error] dataset directory not found: {args.dataset_dir}")
        raise SystemExit(1)

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"[convert] {mode}polygon → bbox  in: {args.dataset_dir}")
    convert_dataset(args.dataset_dir, dry_run=args.dry_run)
    print("[convert] done.")


if __name__ == "__main__":
    main()
