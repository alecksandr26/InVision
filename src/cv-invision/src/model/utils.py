import gdown
from pathlib import Path
from .const import INVISION_DETECTION_MODEL, INVISION_DETECTION_MODEL_CURRENT_VERSION

from ..utils.log import get_logger

logger = get_logger(__name__)


def download_detection_model(version: str = INVISION_DETECTION_MODEL_CURRENT_VERSION, dest_path: str = None):
    """
    Downloads the Invision Detection Model from Google Drive using gdown.

    Args:
        version  : Model version to download (e.g. "v1.0.0"). Defaults to current version.
        dest_path: Local path to save the model. Defaults to model/weights/<filename>.
    """

    model = INVISION_DETECTION_MODEL[version]
    url   = model["url"]
    dest  = str(dest_path or model["filename"])

    logger.info(f"Downloading Invision Detection Model {version} ({model['architecture']} | {model['train_run']})...")

    gdown.download(url, dest)

    logger.info(f"✅ Detection model {version} saved to '{dest}'")


def print_device_info(device: str):
    import torch
    
    logger.info(f"⚙️  Torch: {torch.__version__}")
    logger.info(f"⚙️  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name   = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"⚙️  GPU {i}: {name} ({memory:.1f} GB)")
    else:
        logger.info("⚙️  No GPU detected — running on CPU")

    logger.info(f"⚙️  Selected device: {device.upper()}")


def download_quantized_model(version: str = INVISION_DETECTION_MODEL_CURRENT_VERSION, dest_path: str = None):
    """
    Downloads the Quantized TFLite Invision Detection Model.
    Ensures the file is saved with a .tflite extension to prevent loading errors.
    """
    model = INVISION_DETECTION_MODEL[version]
    
    url = model.get("quantize_url")
    if not url:
        logger.error(f"❌ No quantized model URL found for version {version}")
        return

    # Use the specific quantized_filename from constants if available
    if dest_path:
        dest = Path(dest_path)
    elif "quantized_filename" in model:
        dest = Path(model["quantized_filename"])
    else:
        # Fallback logic: force .tflite extension
        original_path = Path(model["filename"])
        dest = original_path.with_name(f"{original_path.stem}_quantized.tflite")

    logger.info(f"Downloading Quantized Invision Model {version} to {dest}...")
    
    # Download the file
    gdown.download(url, str(dest))
    
    logger.info(f"✅ Quantized TFLite model {version} saved to '{dest}'")


    
if __name__ == "__main__":
    # Download current version
    download_detection_model()
    download_quantized_model()

    # Or download a specific version explicitly:
    # download_detection_model(version="v1.0.0")
    # download_detection_model(version="v1.0.0", dest_path="models/invision_v1.pt")


    
