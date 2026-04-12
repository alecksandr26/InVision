from .const import DETECTION_CONF_TEST, DETECTION_IOU_TEST, DETECTION_CONF_PROD, DETECTION_IOU_PROD, \
    INVISION_DETECTION_MODEL_CURRENT_VERSION, INVISION_DETECTION_MODEL

from .utils import print_device_info
from ..utils.log import get_logger

logger = get_logger(__name__)


def load_detection_model_test(version: str = INVISION_DETECTION_MODEL_CURRENT_VERSION):
    """
    Loads the detection model for TESTING purposes.
    - Lower confidence threshold to catch more detections
    - No GPU enforced, runs on CPU fine
    - Useful for quick sanity checks and debugging

    Args:
        version: Model version to load (e.g. "v1.0.0"). Defaults to current version.

    Returns:
        model: YOLOv5 model instance ready for inference
    """
    import torch 
    from ultralytics import YOLO

    
    model_info = INVISION_DETECTION_MODEL[version]
    weights    = str(model_info["filename"])
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"🧪 Loading Detection Model {version} in TEST mode ({model_info['architecture']} | {model_info['train_run']})...")
    print_device_info(device)

    model                   = YOLO(weights)
    model.overrides["conf"] = DETECTION_CONF_TEST
    model.overrides["iou"]  = DETECTION_IOU_TEST
    model.to(device)

    logger.info(f"✅ Test model loaded from '{weights}' on {device.upper()}")
    return model


def load_detection_model_prod(version: str = INVISION_DETECTION_MODEL_CURRENT_VERSION):
    """
    Loads the detection model for PRODUCTION purposes.
    Automatically exports to TFLite for CPU efficiency if on a Pi/CPU.
    """
    import torch 
    from ultralytics import YOLO
    from pathlib import Path

    model_info = INVISION_DETECTION_MODEL[version]
    weights_path = Path(model_info["filename"])
    
    # Check if we are running on a CPU-only environment (like the Pi)
    # On your Arch machine with the RX 5500 XT, it will use CUDA + .pt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"🚀 Loading Detection Model {version} in PROD mode...")

    # OPTIMIZATION: If on CPU, we want to use TFLite
    if device == "cpu":
        # TFLite export creates a directory with the model inside
        tflite_path = weights_path.with_suffix('.tflite')
        
        if not tflite_path.exists():
            logger.info("⚙️ CPU detected. Exporting to TFLite for better performance...")
            # We load the .pt just to export it
            temp_model = YOLO(str(weights_path))
            temp_model.export(format="tflite", int8=False) # int8=True needs a calibration dataset
            logger.info("✅ Export complete.")

        # Load the optimized version
        model = YOLO(str(tflite_path), task='detect')
    else:
        # On your Arch GPU, the .pt is already super fast
        model = YOLO(str(weights_path))

    # Standard overrides
    model.overrides["conf"] = DETECTION_CONF_PROD
    model.overrides["iou"] = DETECTION_IOU_PROD
    model.to(device)

    # WARM-UP: Essential for real-time response
    import numpy as np
    logger.info("🔥 Warming up model...")
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    logger.info(f"✅ Prod model loaded on {device.upper()}")
    return model


if __name__ == "__main__":
    model = load_detection_model_prod()

    

