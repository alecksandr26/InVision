# =============================================================================
# convert_datasets.py
# Runs polygon → bbox conversion on all datasets flagged in POLYGON_DATASETS.
#
# Prerequisites:
#   - Run extract_datasets.py first
#
# Parallelism:
#   Label file conversion is CPU-bound (text parsing + rewriting thousands of
#   files). ProcessPoolExecutor is used so each worker gets its own CPU core
#   and bypasses the GIL entirely. Each worker converts one label file.
#   convert_polygons_to_bbox.py functions are imported directly — no subprocess
#   overhead.
#
# Usage:
#   python scripts/convert_datasets.py
#   python scripts/convert_datasets.py --dry-run
#   python scripts/convert_datasets.py --workers 4
# =============================================================================

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from const import EXTRACTED_DIR, POLYGON_DATASETS, SPLITS
from convert_polygons_to_bbox import convert_label_file

DEFAULT_WORKERS = min(8, (os.cpu_count() or 4))


# =============================================================================
# Worker (runs in a subprocess — must be a top-level importable function)
# =============================================================================

def _convert_file_worker(args: tuple[Path, bool]) -> tuple[Path, int]:
    """
    Worker function executed in a subprocess.
    Converts one label file in-place and returns (path, num_converted).
    Unpacks a tuple so ProcessPoolExecutor can pickle it cleanly.
    """
    label_path, dry_run = args
    n = convert_label_file(label_path, dry_run=dry_run)
    return label_path, n


# =============================================================================
# Per-dataset converter
# =============================================================================

def convert_dataset_parallel(dataset_dir: Path, dry_run: bool, max_workers: int) -> None:
    """
    Collect all label files across all splits for one dataset,
    then convert them in parallel using a process pool.
    """
    label_files: list[Path] = []

    for split in SPLITS:
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            print(f"  [warn] missing: {dataset_dir.name}/{split}/labels")
            continue
        label_files.extend(sorted(labels_dir.glob("*.txt")))

    if not label_files:
        print(f"  [warn] no label files found in {dataset_dir.name}")
        return

    total_files     = len(label_files)
    total_converted = 0
    errors          = 0

    print(f"  {total_files} label files — {max_workers} workers")

    tasks = [(lf, dry_run) for lf in label_files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_convert_file_worker, task): task[0] for task in tasks}

        for future in as_completed(futures):
            try:
                _, n = future.result()
                total_converted += n
            except Exception as exc:
                errors += 1
                print(f"  [error] {futures[future].name}: {exc}")

    tag = "[dry-run] " if dry_run else ""
    print(
        f"  {tag}converted {total_converted} polygon annotation(s) "
        f"across {total_files} files "
        f"({'no errors' if not errors else f'{errors} errors'})"
    )


# =============================================================================
# Main
# =============================================================================

def convert_polygon_datasets(dry_run: bool, max_workers: int) -> None:
    """Run parallel polygon → bbox conversion on every POLYGON_DATASETS entry."""
    for zip_path in POLYGON_DATASETS:
        dataset_dir = EXTRACTED_DIR / zip_path.stem

        if not dataset_dir.exists():
            print(f"[error] dataset not extracted yet: {dataset_dir.name}")
            print("        run extract_datasets.py first")
            sys.exit(1)

        print(f"[convert] polygon → bbox  in {dataset_dir.name}/")
        convert_dataset_parallel(dataset_dir, dry_run=dry_run, max_workers=max_workers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert polygon labels → bboxes for all flagged datasets."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be converted without writing any files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Parallel worker processes (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  cv-vision — Convert Polygon Labels")
    print("=" * 60)
    print(f"\n  Workers: {args.workers}\n")

    convert_polygon_datasets(dry_run=args.dry_run, max_workers=args.workers)

    print("\n✅  Done.")


if __name__ == "__main__":
    main()
