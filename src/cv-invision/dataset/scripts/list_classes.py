# =============================================================================
# list_classes.py
# Lists all classes from each extracted dataset's data.yaml.
#
# Prerequisites:
#   - Run extract_datasets.py first
#
# Usage:
#   python scripts/list_classes.py
# =============================================================================

import sys
import yaml

from const import ALL_DATASET_ZIPS, EXTRACTED_DIR


def list_classes() -> None:
    """
    Read data.yaml from each extracted dataset and print their class names.
    Errors out if any dataset hasn't been extracted yet.
    """
    # Check all datasets are extracted before doing anything
    missing = [
        zip_path for zip_path in ALL_DATASET_ZIPS
        if not (EXTRACTED_DIR / zip_path.stem).exists()
    ]

    if missing:
        print("[error] the following datasets have not been extracted yet:")
        for zip_path in missing:
            print(f"        • {zip_path.name}")
        print("\n        run extract_datasets.py first")
        sys.exit(1)

    # Print classes per dataset
    all_classes: set[str] = set()

    for zip_path in ALL_DATASET_ZIPS:
        dataset_dir = EXTRACTED_DIR / zip_path.stem
        yaml_path = dataset_dir / "data.yaml"

        if not yaml_path.exists():
            print(f"[warn] no data.yaml found in {dataset_dir.name}")
            continue

        with open(yaml_path) as f:
            meta = yaml.safe_load(f)

        classes: list[str] = meta.get("names", [])
        all_classes.update(classes)

        print(f"\n── {dataset_dir.name}")
        print(f"   {len(classes)} class(es):")
        for i, cls in enumerate(classes):
            print(f"   [{i}] {cls}")

    # Summary
    print(f"\n── Total unique classes across all datasets: {len(all_classes)}")
    for cls in sorted(all_classes):
        print(f"   • {cls}")


def main() -> None:
    print("=" * 60)
    print("  cv-vision — Dataset Class List")
    print("=" * 60)

    list_classes()


if __name__ == "__main__":
    main()
