import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # ← force xcb display backend

import cv2
import argparse
import numpy as np
from pathlib import Path

from ..detection_model import load_detection_model_test
from ..const            import CLASS_NAMES
from src.utils.log import get_logger

logger = get_logger(__name__)

# ============================================================
# BBOX COLORS — one per class, consistent across runs
# ============================================================
COLORS = [
    (255,  56,  56),  # person        — red
    ( 56, 152, 255),  # vehicle       — blue
    (255, 157,  56),  # obstacle      — orange
    (100, 100, 255),  # pothole       — purple
    ( 56, 255, 122),  # pole          — green
    (255, 255,  56),  # stairs        — yellow
    ( 56, 255, 255),  # crosswalk     — cyan
    (255,  56, 193),  # door          — pink
    (173, 255,  47),  # traffic_light — lime
    (255, 128,   0),  # edge_hazard   — deep orange
]


def get_color(class_id: int) -> tuple:
    """Returns a consistent BGR color for a given class id."""
    return COLORS[class_id % len(COLORS)]


def render_bboxes(image: np.ndarray, results) -> np.ndarray:
    """
    Renders bounding boxes and class labels on the image.

    Args:
        image  : Original BGR image as numpy array
        results: YOLO inference results object

    Returns:
        np.ndarray: Annotated image with bboxes and labels drawn
    """
    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Extract bbox coords
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf            = float(box.conf[0])
            class_id        = int(box.cls[0])
            class_name      = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
            color           = get_color(class_id)
            label           = f"{class_name} {conf:.2f}"

            # Draw bbox rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                image,
                (x1, y1 - label_h - baseline - 4),
                (x1 + label_w, y1),
                color,
                thickness=-1  # filled
            )

            # Draw label text
            cv2.putText(
                image, label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale  = 0.5,
                color      = (0, 0, 0),  # black text on colored background
                thickness  = 1,
                lineType   = cv2.LINE_AA
            )

            logger.debug(f"  [{class_name}] conf={conf:.2f} bbox=({x1},{y1},{x2},{y2})")

    return image


def run_inference(image_path: Path, model, save: bool = True, show: bool = True) -> Path:
    """
    Runs detection inference on a single image and renders the results.

    Args:
        image_path: Path to the input image file
        model     : Loaded YOLO model instance
        save      : Whether to save the annotated image to disk
        show      : Whether to display the annotated image with cv2.imshow()

    Returns:
        Path to the saved output image (if save=True)
    """
    logger.info(f"Running inference on '{image_path.name}'...")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not read image: '{image_path}'")
        raise FileNotFoundError(f"Image not found: '{image_path}'")

    # Run inference
    results = model(image)  # ← pass numpy array directly, not string path

    # Count detections
    total = sum(len(r.boxes) for r in results)
    logger.info(f"Detected {total} object(s) in '{image_path.name}'")

    # Render bboxes on image
    annotated = render_bboxes(image.copy(), results)

    # Save output
    output_path = None
    if save:
        output_path = image_path.parent / f"{image_path.stem}_result{image_path.suffix}"
        cv2.imwrite(str(output_path), annotated)
        logger.info(f"✅ Saved result to '{output_path}'")

    # Display output
    if show:
        window_name = f"InVision Detection — {image_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)   # ← resizable window
        cv2.resizeWindow(window_name, 800, 800)            # ← set initial size
        cv2.imshow(window_name, annotated)
        logger.info("🖼️  Press 'Q' to close the window...")

        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    return output_path

    
def parse_args():
    parser = argparse.ArgumentParser(
        description="🔍 InVision Detection Model — Single Image Inference"
    )
    parser.add_argument(
        "--input", "-i",
        type     = str,
        required = True,
        help     = "Path to input image file (e.g. src/data-examples/test.jpg)"
    )
    parser.add_argument(
        "--no-save",
        action  = "store_true",
        default = False,
        help    = "Do not save the annotated output image"
    )
    parser.add_argument(
        "--no-show",
        action  = "store_true",
        default = False,
        help    = "Do not display the annotated image window"
    )
    return parser.parse_args()


def main():
    args       = parse_args()
    image_path = Path(args.input)

    if not image_path.exists():
        logger.error(f"Input image not found: '{image_path}'")
        raise SystemExit(1)

    logger.info("🚀 Starting InVision inference pipeline...")

    # Load model
    model = load_detection_model_test()

    # Run inference
    run_inference(
        image_path = image_path,
        model      = model,
        save       = not args.no_save,
        show       = not args.no_show,
    )


if __name__ == "__main__":
    main()
