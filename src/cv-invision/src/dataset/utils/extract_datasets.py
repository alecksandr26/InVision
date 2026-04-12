# =============================================================================
# extract_datasets.py
# Extracts all raw dataset zips defined in const.py into EXTRACTED_DIR.
#
# Parallelism:
#   Each zip is extracted in its own thread. Since zipfile extraction is
#   I/O-bound (disk reads + writes) and involves zlib decompression that
#   releases the GIL, ThreadPoolExecutor gives real speedup here.
#   One thread per zip — no point in more workers than datasets.
#
# Usage:
#   python scripts/extract_datasets.py
#   python scripts/extract_datasets.py --workers 2
# =============================================================================

import argparse
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from const import ALL_DATASET_ZIPS, EXTRACTED_DIR


# =============================================================================
# Worker
# =============================================================================

def extract_one(zip_path: Path) -> tuple[str, str]:
    """
    Extract a single zip into EXTRACTED_DIR/<zip_stem>/.
    Returns (zip name, status message) for reporting.
    Raises on any error so the executor can catch and surface it.
    """
    out_dir = EXTRACTED_DIR / zip_path.stem

    if out_dir.exists():
        return zip_path.name, f"[skip]    already extracted → {out_dir.name}"

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    return zip_path.name, f"[extract] {zip_path.name} → {out_dir.name}/"


# =============================================================================
# Main logic
# =============================================================================

def extract_all(max_workers: int) -> None:
    """Validate zips exist, then extract all in parallel."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    # Fail fast — check all zips exist before spawning threads
    missing = [z for z in ALL_DATASET_ZIPS if not z.exists()]
    if missing:
        for z in missing:
            print(f"[error] zip not found: {z}")
        sys.exit(1)

    workers = min(max_workers, len(ALL_DATASET_ZIPS))
    print(f"  Workers: {workers}  |  Zips: {len(ALL_DATASET_ZIPS)}\n")

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for zip_path in ALL_DATASET_ZIPS:
            future = executor.submit(extract_one, zip_path)
            futures[future] = zip_path.name

    # Print results in completion order
    for future in as_completed(futures):
        try:
            _, msg = future.result()
            print(msg)
        except Exception as exc:
            print(f"[error] {futures[future]}: {exc}")
            sys.exit(1)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all raw dataset zips in parallel."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=len(ALL_DATASET_ZIPS),
        metavar="N",
        help=f"Max parallel extraction threads (default: {len(ALL_DATASET_ZIPS)}, one per zip)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  cv-vision — Extract Datasets")
    print("=" * 60)
    print()
    extract_all(max_workers=args.workers)
    print(f"\n✅  Done. Extracted datasets → {EXTRACTED_DIR}")


if __name__ == "__main__":
    main()
