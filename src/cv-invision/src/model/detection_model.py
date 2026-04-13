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
    - GPU (CUDA): Loads standard .pt directly.
    - CPU (Edge/Pi): Prefers the pre-downloaded .tflite quantized model.
    """
    import torch
    import numpy as np
    from ultralytics import YOLO
    from pathlib import Path

    model_info   = INVISION_DETECTION_MODEL[version]
    weights_path = Path(model_info["filename"])
    
    # Updated to look for the .tflite version specifically
    quantized_path = model_info.get("quantized_filename") or \
                     weights_path.with_name(f"{weights_path.stem}_quantized.tflite")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"🚀 Loading Detection Model {version} in PROD mode...")

    # ── CPU/Edge Optimization (Raspberry Pi Path) ──────────────────
    if device == "cpu":
        if Path(quantized_path).exists():
            logger.info(f"📦 Found pre-downloaded quantized TFLite model: {quantized_path}")
            # We don't use torch.load here because it's a TFLite file!
            model = YOLO(str(quantized_path), task="detect")
        else:
            logger.warning(f"⚠️  Quantized TFLite not found at {quantized_path}. Falling back to standard weights.")
            model = YOLO(str(weights_path))

        # Define specialized predict for CPU
        def predict(source, **kwargs):
            return model.predict(
                source  = source,
                conf    = DETECTION_CONF_PROD,
                iou     = DETECTION_IOU_PROD,
                verbose = kwargs.get("verbose", False),
            )

        logger.info(f"✅ Prod model loaded on CPU (Optimized with TFLite)")
        return predict

    # ── GPU Path (Your RX 5500 XT) ────────────────────────────────
    logger.info(f"📦 Loading standard .pt weights from '{weights_path}'...")
    model = YOLO(str(weights_path))
    model.overrides["conf"] = DETECTION_CONF_PROD
    model.overrides["iou"]  = DETECTION_IOU_PROD
    model.to(device)

    # Warmup
    logger.info("🔥 Warming up model...")
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    logger.info(f"✅ Prod model loaded on {device.upper()}")

    return model


if __name__ == "__main__":
    model = load_detection_model_prod()

    

