from pathlib import Path
from datetime import datetime


from ..model.const import DETECTION_INPUT_IMAGE_SIZE

# ============================================================
# TEST SAMPLES - RAW SAMPLE MEDIA
# ============================================================

# Sample media (mp4 videos and/or images) hosted on Google Drive
# used for testing and validating the InVision detection model inference pipeline.
# These are NOT training/dataset files — just raw test clips and images.
TEST_SAMPLES_DIR_URL  = "https://drive.google.com/drive/folders/1OP7Bw-ltSF3tAYHJhGTe4NTesufE1Ibi?usp=drive_link"

# Local directory where test samples will be downloaded
TEST_SAMPLES_DIR = Path(__file__).parent.parent / "data-examples"
TEST_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SUPPORTED MEDIA EXTENSIONS
# ============================================================

# Supported image file extensions for the InVision detection pipeline.
# These are the formats accepted as input for running inference.
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Supported video file extensions for the InVision detection pipeline.
# These are the formats accepted as input for frame extraction and inference.
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

# ============================================================
# MEDIA PROCESSING CONFIG
# ============================================================

# Output image quality when saving processed frames or results (0-100).
# 85 is a good balance between visual quality and file size.
IMAGE_QUALITY = 85

# Input size for resizing media before running detection inference.
# Imported from model/const.py — must match the training image size (640x640).
INPUT_IMAGE_SIZE = DETECTION_INPUT_IMAGE_SIZE



# ============================================================
# INVISION LOGGER SETUP
# ============================================================

# Log files will be stored in logs/ at the project root
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Log filename with timestamp so each session has its own file
LOG_FILE = LOGS_DIR / f"invision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
