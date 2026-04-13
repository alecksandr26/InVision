from pathlib import Path

from ..dataset.utils.const import UNIFIED_CLASSES


# ============================================================
# DETECTION MODEL CLASSES
# ============================================================

# Class names used by the detection model — imported from the dataset pipeline.
# Index position matches the class id in the label files.
# Used for rendering bboxes and class labels during inference.
CLASS_NAMES: list[str] = UNIFIED_CLASSES


# ============================================================
# PATHS
# ============================================================
MODEL_DIR     = Path(__file__).parent          # .../cv-vision/model/
WEIGHTS_DIR   = MODEL_DIR / "weights"          # .../cv-vision/model/weights/
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)  # ← create it right here on import


# ============================================================
# INVISION DETECTION MODELS - VERSION REGISTRY
# ============================================================

INVISION_DETECTION_MODEL = {
    "v1.0.0": {
        "architecture": "YOLOv5n",
        "train_run":    "train6",
        "dataset":      "invision-dataset-v1",
        "description":  "First trained model from our custom dataset using YOLOv5n",
        "url":          "https://drive.google.com/file/d/1BUs5CwDfT1d4B3TFK3HByq1Y2n9UPblM/view?usp=sharing",
        "quantize_url": "https://drive.google.com/file/d/1RO6Y7Mc3hueuaNVsPVoVLT3EIGmFh4es/view?usp=sharing",
        "filename":     WEIGHTS_DIR / "invision_detection_v1.pt",
        "quantized_filename": WEIGHTS_DIR / "invision_detection_v1_quantized.tflite",
    },

    # "v2.0.0": {
    #     "architecture": "YOLOv5s",
    #     "train_run":    "train12",
    #     "dataset":      "invision-dataset-v2",
    #     "description":  "Improved model with larger backbone and more data",
    #     "url":          "https://drive.google.com/file/d/ANOTHER_FILE_ID/view?usp=sharing",
    #     "filename":     WEIGHTS_DIR / "invision_detection_v2.pt",
    # },
}

# --- Active version pointer ---
INVISION_DETECTION_MODEL_CURRENT_VERSION = "v1.0.0"


# ============================================================
# DETECTION MODEL INFERENCE PARAMS - TEST
# ============================================================
DETECTION_CONF_TEST = 0.25  # lower threshold — catch more detections, good for debugging
DETECTION_IOU_TEST  = 0.45  # overlap threshold for NMS (non-max suppression)

# ============================================================
# DETECTION MODEL INFERENCE PARAMS - PROD
# ============================================================
DETECTION_CONF_PROD = 0.60  # higher threshold — only reliable detections
DETECTION_IOU_PROD  = 0.45  # same NMS threshold, can tune independently if needed

# ============================================================
# DETECTION MODEL INPUT CONFIG
# ============================================================

# Input image size for the detection model (width x height in pixels).
# YOLOv5n was trained with 640x640 images — changing this may affect accuracy.
# This value is also used for video scaling in preprocessing pipelines.
DETECTION_INPUT_IMAGE_SIZE = 640


# ============================================================
# DEEP SORT TRACKER CONSTANTS
# ============================================================

# Frames to keep a lost track alive before deleting
TRACKER_MAX_AGE = 6            # was 5 — too aggressive, lost tracks too fast

# Frames needed to confirm a track
TRACKER_N_INIT = 5              # was 7 — way too strict, missed most detections

# NMS overlap threshold
TRACKER_NMS_MAX_OVERLAP = 0.60   # was 0.15 — too aggressive, killed valid detections

# Cosine distance for appearance matching
TRACKER_MAX_COSINE_DISTANCE = 0.5  # keep this one, it's fine

# Gallery size
TRACKER_NN_BUDGET = 100         # was 50, give it a bit more room

# FP16
TRACKER_HALF = True

# BGR format
TRACKER_BGR = True

# Embedder
TRACKER_EMBEDDER = "mobilenet"

# Min confidence to feed into tracker
TRACKER_MIN_CONF = 0.4          # was 0.6 — too strict, dropped valid detections

