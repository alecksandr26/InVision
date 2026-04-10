# =============================================================================
# const.py
# Central configuration for the cv-vision dataset pipeline.
#
# All datasets are Roboflow exports in YOLOv5 PyTorch format.
#
# Pipeline order:
#   1. extract_datasets.py          — unzip raw datasets
#   2. convert_datasets.py          — polygon → bbox  (SafeWalkBD)
#   3. remap_classes.py             — N raw classes → 10 unified classes
#   4. merge_datasets.py            — merge splits + rebuild data.yaml
#   5. build.py                     — final validation / build
# =============================================================================

from pathlib import Path

# =============================================================================
# Root Paths
# =============================================================================

# Absolute root of the data directory
DATA_DIR = Path(__file__).parent.parent  # .../cv-vision/data

# Raw zip files (Roboflow exports)
RAW_DIR = DATA_DIR / "raw"

# Extracted datasets (intermediate, created during pipeline)
EXTRACTED_DIR = DATA_DIR / "extracted"

# Final merged + cleaned dataset
MERGED_DIR = DATA_DIR / "merged"


# =============================================================================
# Raw Dataset Zips
# =============================================================================

# Roboflow: pedestrian + obstacle detection
# Labels: bounding boxes — ready to use
PEDESTRIAN_OBSTACLE_DETECTION_ZIP = RAW_DIR / "PedestrianObstacleDetection.v1i.yolov5pytorch.zip"

# Roboflow: SafeWalkBD dataset (v9)
# ⚠️  WARNING: Labels use POLYGON annotations, NOT bounding boxes.
#     Must run convert_datasets.py on this dataset BEFORE remapping.
#     Algorithm: axis-aligned bbox from polygon → x_min, y_min, x_max, y_max
#                derived from the min/max extent of all polygon vertices.
SAFE_WALK_BD_ZIP = RAW_DIR / "SafeWalkBD.v9i.yolov5pytorch.zip"

# Roboflow: general vision dataset (v1)
# Labels: bounding boxes — ready to use
VISION_ZIP = RAW_DIR / "vision.v1i.yolov5pytorch.zip"


# =============================================================================
# Dataset Registry
# =============================================================================

# All zips in pipeline order
ALL_DATASET_ZIPS = [
    PEDESTRIAN_OBSTACLE_DETECTION_ZIP,
    SAFE_WALK_BD_ZIP,
    VISION_ZIP,
]

# Datasets that require polygon → bbox conversion before anything else
POLYGON_DATASETS = [
    SAFE_WALK_BD_ZIP,
]

# Standard YOLOv5 dataset splits
SPLITS = ["train", "valid", "test"]


# =============================================================================
# Unified Classes
# 10 classes tailored for blind-person outdoor navigation.
# Index position = class id used in label files.
# =============================================================================

UNIFIED_CLASSES: list[str] = [
    "person",           # 0 — pedestrians, animals (moving agents near the user)
    "vehicle",          # 1 — cars, bikes, buses, trucks, trains (fast-moving hazards)
    "obstacle",         # 2 — generic blocking objects: bins, chairs, cones, barriers…
    "pothole",          # 3 — ground hazards: potholes, puddles, drains, bad road
    "pole",             # 4 — vertical fixed structures
    "stairs",           # 5 — elevation changes: stairs, overpasses
    "crosswalk",        # 6 — navigation guidance: zebra crossings, tactile paving
    "door",             # 7 — entry/exit points: open or closed doors
    "traffic_light",    # 8 — traffic lights, signals, road signs
    "edge_hazard",      # 9 — drop boundaries: fences, railings, sidewalk edges
]

# Quick lookup: class name → integer id
UNIFIED_CLASS_ID: dict[str, int] = {cls: i for i, cls in enumerate(UNIFIED_CLASSES)}

# Number of classes (used when writing data.yaml)
NUM_CLASSES: int = len(UNIFIED_CLASSES)


# =============================================================================
# Class Remap Table
# Maps every raw dataset class name (lowercase) → unified class name.
# Used by remap_classes.py to rewrite label files.
#
# Classes not present in this map are considered DROPPED — their annotation
# lines are removed and images with no remaining labels are discarded.
# =============================================================================

CLASS_REMAP: dict[str, str] = {

    # ── person ────────────────────────────────────────────────────────────────
    "animal":               "person",   # SafeWalkBD
    "person":               "person",   # SafeWalkBD / vision
    "pedestrian":           "person",   # PedestrianObstacleDetection

    # ── vehicle ───────────────────────────────────────────────────────────────
    "car":                  "vehicle",  # PedestrianObstacleDetection / vision
    "motorcycle":           "vehicle",  # PedestrianObstacleDetection
    "motorbike":            "vehicle",  # vision
    "bus":                  "vehicle",  # vision
    "truck":                "vehicle",  # vision
    "tricycle":             "vehicle",  # vision
    "bicycle":              "vehicle",  # vision
    "vehicle":              "vehicle",  # SafeWalkBD
    "train":                "vehicle",  # SafeWalkBD

    # ── obstacle ──────────────────────────────────────────────────────────────
    "obstacle":             "obstacle", # PedestrianObstacleDetection / vision
    "roadblock":            "obstacle", # PedestrianObstacleDetection
    "spherical_roadblock":  "obstacle", # vision
    "reflective_cone":      "obstacle", # vision
    "warning_column":       "obstacle", # vision
    "gate barrier":         "obstacle", # PedestrianObstacleDetection
    "road-barrier":         "obstacle", # SafeWalkBD
    "garbage bin":          "obstacle", # PedestrianObstacleDetection
    "chair":                "obstacle", # PedestrianObstacleDetection
    "plant pot":            "obstacle", # PedestrianObstacleDetection
    "ashcan":               "obstacle", # vision
    "street vendor":        "obstacle", # PedestrianObstacleDetection
    "fire_hydrant":         "obstacle", # vision

    # ── pothole ───────────────────────────────────────────────────────────────
    "pothole":              "pothole",  # PedestrianObstacleDetection / SafeWalkBD
    "puddle":               "pothole",  # PedestrianObstacleDetection
    "drain":                "pothole",  # PedestrianObstacleDetection
    "bad road":             "pothole",  # PedestrianObstacleDetection

    # ── pole ──────────────────────────────────────────────────────────────────
    "pole":                 "pole",     # PedestrianObstacleDetection / SafeWalkBD / vision

    # ── stairs ────────────────────────────────────────────────────────────────
    "stair":                "stairs",   # PedestrianObstacleDetection
    "stairs":               "stairs",   # SafeWalkBD / vision
    "over-bridge":          "stairs",   # SafeWalkBD

    # ── crosswalk ─────────────────────────────────────────────────────────────
    "zebra cross":          "crosswalk",# PedestrianObstacleDetection
    "zebra_crossing":       "crosswalk",# vision
    "crosswalk":            "crosswalk",# SafeWalkBD
    "tactile_paving":       "crosswalk",# vision
    "sidewalk":             "crosswalk",# SafeWalkBD

    # ── door ──────────────────────────────────────────────────────────────────
    "door":                 "door",     # PedestrianObstacleDetection
    "close_door":           "door",     # vision
    "open_door":            "door",     # vision

    # ── traffic_light ─────────────────────────────────────────────────────────
    "traffic_lights":       "traffic_light", # vision
    "traffic-light":        "traffic_light", # SafeWalkBD
    "human_traffic_lights": "traffic_light", # vision
    "traffic-sign":         "traffic_light", # SafeWalkBD
    "traffic_sign":         "traffic_light", # SafeWalkBD (alt key)
    "stop_sign":            "traffic_light", # vision
    "street_signs":         "traffic_light", # vision

    # ── edge_hazard ───────────────────────────────────────────────────────────
    "fence":                "edge_hazard",   # PedestrianObstacleDetection
    "railing":              "edge_hazard",   # vision

}

# =============================================================================
# Dropped Classes
# Raw class names explicitly excluded from the unified dataset.
# Kept here for traceability — so it's clear these were a conscious decision,
# not an oversight.
# =============================================================================

DROPPED_CLASSES: dict[str, str] = {
    "left turn":        "directional sign — not an obstacle",
    "right turn":       "directional sign — not an obstacle",
    "ticket_barrier":   "rare indoor context, low navigation value",
    "railway":          "static track — train itself is captured as vehicle",
}


# Config for the clean and resize script
TARGET_SIZE = 640
MIN_DIM = 200
