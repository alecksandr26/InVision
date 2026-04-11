import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Import InVision configurations
import const

def load_labels(split_dir: Path) -> pd.DataFrame:
    """Reads all YOLO .txt label files into a single Pandas DataFrame."""
    labels_dir = split_dir / "labels"
    if not labels_dir.exists():
        return pd.DataFrame()

    all_data = []
    
    # YOLO format: class_id x_center y_center width height
    for txt_file in tqdm(list(labels_dir.glob("*.txt")), desc=f"Reading {split_dir.name}"):
        try:
            # Load text file into numpy array
            data = np.loadtxt(str(txt_file), ndmin=2)
            if data.size > 0:
                # Add a column for the filename to track objects per image
                img_name = txt_file.stem
                for row in data:
                    all_data.append([img_name, int(row[0]), row[1], row[2], row[3], row[4]])
        except Exception:
            pass # Skip empty or corrupt files

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["image", "class_id", "x_center", "y_center", "width", "height"])
    return df

def analyze_and_plot(df: pd.DataFrame, output_path: str = "dataset_dashboard.png"):
    """Generates the 4-panel analysis dashboard."""
    print(f"\nTotal bounding boxes: {len(df):,}")
    
    # Map class IDs to the actual names from const.py
    df["class_name"] = df["class_id"].map(lambda x: const.UNIFIED_CLASSES[x] if x < len(const.UNIFIED_CLASSES) else "unknown")

    # Set up the Matplotlib figure (2x2 grid)
    plt.style.use('dark_background') # Looks great for computer vision dashboards
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Project InVision: Dataset Analysis Dashboard", fontsize=20, fontweight="bold")

    # --- 1. Class Distribution (Top Left) ---
    class_counts = df["class_name"].value_counts()
    axs[0, 0].bar(class_counts.index, class_counts.values, color='cornflowerblue')
    axs[0, 0].set_title("Class Distribution (Imbalance Check)")
    axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 0].set_ylabel("Number of Instances")

    # --- 2. Bounding Box Size Distribution (Top Right) ---
    # Convert normalized width/height to percentages for easier reading
    axs[0, 1].scatter(df["width"] * 100, df["height"] * 100, alpha=0.1, s=1, color='cyan')
    axs[0, 1].set_title("Bounding Box Scales (Relative to Image %)")
    axs[0, 1].set_xlabel("Width %")
    axs[0, 1].set_ylabel("Height %")
    axs[0, 1].set_xlim(0, 100)
    axs[0, 1].set_ylim(0, 100)

    # --- 3. Objects Per Image (Bottom Left) ---
    objects_per_img = df.groupby("image").size()
    axs[1, 0].hist(objects_per_img, bins=range(0, int(objects_per_img.max()) + 5), color='mediumspringgreen', edgecolor='black')
    axs[1, 0].set_title("Spatial Density (Objects per Image)")
    axs[1, 0].set_xlabel("Number of Objects")
    axs[1, 0].set_ylabel("Image Count")

    # --- 4. Spatial Heatmap (Bottom Right) ---
    # Where do objects appear on the screen? (Normalized coordinates 0 to 1)
    hb = axs[1, 1].hexbin(df["x_center"], df["y_center"], gridsize=40, cmap='inferno', mincnt=1)
    axs[1, 1].invert_yaxis() # Invert Y so it matches image coordinates (0 at top)
    axs[1, 1].set_title("Spatial Center Heatmap (Location Bias)")
    axs[1, 1].set_xlabel("X Center")
    axs[1, 1].set_ylabel("Y Center")
    fig.colorbar(hb, ax=axs[1, 1], label="Density")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n[+] Dashboard saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze YOLO dataset statistics.")
    args = parser.parse_args()

    print("=" * 60)
    print("  cv-vision — Dataset Analyzer")
    print("=" * 60)

    frames = []
    for split in const.SPLITS:
        split_dir = const.MERGED_DIR / split
        if split_dir.exists():
            df = load_labels(split_dir)
            if not df.empty:
                frames.append(df)

    if not frames:
        print("[error] No label data found. Ensure labels are in the merged directory.")
        return

    full_df = pd.concat(frames, ignore_index=True)
    analyze_and_plot(full_df)

if __name__ == "__main__":
    main()
