# Project InVision: Dataset Pipeline

This repository contains the core pipeline for building the unified **InVision Dataset**. It automates the extraction, cleaning, remapping, and merging of multiple Roboflow sources into a single, high-quality dataset optimized for YOLOv5 training.

## 📥 Dataset Sources
We aggregate and normalize data from the following Roboflow projects:
* [Pedestrian Obstacle Detection](https://universe.roboflow.com/pedestrianobstacle/pedestrianobstacledetection)
* [SafeWalkBD](https://universe.roboflow.com/safewalkbd/safewalkbd-l8jbn)
* [Vision-yjlv3](https://universe.roboflow.com/segmentation-vgo6s/vision-yjlv3)

---

## 🛠 How to Build the Dataset
Follow these steps in order to generate the final `merged/` directory and the distributable `.zip` archive.

### 1. Initial Setup
Place all raw Roboflow exports (YOLOv5 PyTorch format) into the `data/raw/` directory.

### 2. Extraction
Unzip all datasets into the intermediate `extracted/` directory.
```bash
python scripts/extract_datasets.py
````

### 3\. Polygon to Bounding Box Conversion

The **SafeWalkBD** dataset often uses polygons. This script calculates the tightest axis-aligned bounding box for every polygon to ensure compatibility with our detection model.

```bash
python scripts/convert_datasets.py
```

### 4\. Class Remapping & Normalization

We consolidate over 60 raw class names into **10 unified InVision classes**.

*Refer to `const.py` for the specific mapping logic and the `DROPPED_CLASSES` list.*

```bash
python scripts/remap_classes.py
```

### 5\. Final Merge

Merge the splits (train/valid/test) from all datasets and build the global `data.yaml`.

```bash
python scripts/merge_datasets.py
```

### 6\. Clean and Resize (Optimization)

Filter out images smaller than 200px and resize/letterbox everything to **640x640** to ensure consistency for the model.

```bash
python scripts/clean_and_resize.py --workers 8
```

### 7\. Dataset Analysis

Generate the statistical dashboard to verify class distribution, object scales, and spatial bias.

```bash
python scripts/analyze_dataset.py
```

### 8\. Package Build

Create the final distributable `.zip` archive for training (e.g., to upload to Colab or Drive).

```bash
python scripts/build.py --version 1
```

-----

## 📈 V1 Baseline Statistics

The following metrics represent the current state of our merged data:

  * **Class Distribution:** High imbalance (Person/Car dominated). Critical navigation hazards like potholes and stairs are minority classes.
  * **Scale Distribution:** Majority of objects are small (\<10% area); Mosaic augmentation is required during training.
  * **Spatial Heatmap:** Strong center-lower bias; translation augmentation is recommended to improve edge detection.

-----

## 📂 Directory Structure

  * `data/raw/`: Original zip files.
  * `data/extracted/`: Intermediate per-dataset folders.
  * `data/merged/`: The final unified dataset ready for training.
  * `scripts/`: Python implementation of the pipeline stages.
      * `const.py`: Central configuration and class mapping logic.

-----

## 📊 Dataset Analysis & Baseline (V1)

Before starting the training process, we analyze the unified dataset to identify potential biases or imbalances that could affect model performance.

### Class Merging Strategy
We consolidate over 60 raw class names from multiple sources into **10 unified InVision classes**. This mapping ensures the model focuses on the most critical navigation obstacles.

![Class Merging Map](class_merging_map.svg)

### Statistical Dashboard
The dashboard below provides a deep dive into the 25,000+ images currently in the `merged/` directory.

![Dataset Dashboard](dataset_dashboard.png)

#### Key Observations:
* **Class Imbalance:** There is a significant dominant frequency of `person` and `car` labels. Minor classes like `pothole` and `stairs` represent less than 5% of the total annotations. 
    * *Strategy:* We will monitor V1 test results to see if these rare classes require synthetic augmentation (Copy-Paste) in V2.
* **Object Scale:** The scatter plot shows a high concentration of "small" objects (bottom-left). 
    * *Strategy:* Training will utilize **640px resolution** and **Mosaic Augmentation** to prevent these small hazards from being missed.
* **Spatial Bias:** The heatmap reveals a central "hotspot" for object centers. 
    * *Strategy:* We will enable **Translation Augmentation** (0.1) during training to force the model to look at the periphery of the frame.

---
