import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time
import argparse
import threading
import queue
from pathlib import Path

from src.pipeline.factory  import build_tracking_pipeline, build_detection_pipeline
from src.pipeline.sources  import VideoFileSource, CameraSource
from src.pipeline.const    import SENTINEL
from src.model.detection_model import load_detection_model_prod, load_detection_model_test
from src.model.tracker         import build_tracker
from src.utils.log             import get_logger

logger = get_logger(__name__)


def display_output(show: bool = True, save_path: str = None, fps: float = 20.0):
    """
    Returns an output handler function for the RendererStage.
    Handles display and/or saving of annotated frames.

    Args:
        show      : Whether to display frames with cv2.imshow
        save_path : Path to save output video — None to skip
        fps       : FPS for saved video
    """
    writer     = None
    stop_event = threading.Event()

    def handler(data):
        nonlocal writer

        frame = data.annotated
        if frame is None:
            return

        # ── display ───────────────────────────────────────────
        if show:
            cv2.namedWindow("InVision Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("InVision Detection", 800, 800)
            cv2.imshow("InVision Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()

        # ── save ──────────────────────────────────────────────
        if save_path and writer is None:
            h, w   = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
            logger.info(f"💾 Saving output to '{save_path}'")

        if writer:
            writer.write(frame)

    def release():
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    return handler, stop_event, release


def run_pipeline(
    source,
    mode      : str  = "tracking",
    test_mode : bool = True,
    show      : bool = True,
    save_path : str  = None,
):
    logger.info(f"🚀 Starting InVision pipeline...")
    
    model = load_detection_model_test() if test_mode else load_detection_model_prod()

    # ── build pipeline ────────────────────────────────────────
    # Note: factory functions now return (stages, out_queue)
    if mode == "tracking":
        tracker = build_tracker() if test_mode else build_tracker()
        stages, out_queue = build_tracking_pipeline(source, model, tracker)
    else:
        stages, out_queue = build_detection_pipeline(source, model)

    for stage in stages:
        stage.start()

    logger.info(f"✅ Pipeline running — press Q to stop")

    writer = None
    try:
        # ── MAIN THREAD UI LOOP ───────────────────────────────
        while True:
            try:
                # Pull finished data from the final queue
                data = out_queue.get(timeout=0.1)
                
                # Sentinel (None) indicates the source is finished
                if data is None:
                    break

                frame = data.annotated
                
                # Rendering logic (Must happen on Main Thread for macOS)
                if show and frame is not None:
                    cv2.namedWindow("InVision Detection", cv2.WINDOW_NORMAL)
                    cv2.imshow("InVision Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Saving logic
                if save_path and frame is not None:
                    if writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        fps = source.fps if source.fps > 0 else 20.0
                        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                    writer.write(frame)

            except queue.Empty:
                # Check if the pipeline stages have crashed or finished naturally
                if not any(s._thread.is_alive() for s in stages):
                    logger.info("✅ Pipeline stages completed")
                    break

    except KeyboardInterrupt:
        logger.info("⚠️  Interrupted by user")

    finally:
        logger.info("🛑 Stopping pipeline...")
        # Clean shutdown of all threads
        for stage in stages:
            stage.stop()
        for stage in stages:
            stage.join()

        if writer:
            writer.release()
        
        cv2.destroyAllWindows()
        logger.info("✅ Pipeline stopped cleanly")


def parse_args():
    parser = argparse.ArgumentParser(
        description="🚀 InVision Detection Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type    = str,
        default = None,
        help    = "Path to video file or camera device id (default: camera 0)"
    )
    parser.add_argument(
        "--output", "-o",
        type    = str,
        default = None,
        help    = "Path to save annotated output video"
    )
    parser.add_argument(
        "--mode", "-m",
        type    = str,
        default = "tracking",
        choices = ["tracking", "detection"],
        help    = "Pipeline mode (default: tracking)"
    )
    parser.add_argument(
        "--prod",
        action  = "store_true",
        default = False,
        help    = "Use production model (default: test model)"
    )
    parser.add_argument(
        "--no-show",
        action  = "store_true",
        default = False,
        help    = "Do not display output window"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── build source ──────────────────────────────────────────
    if args.input is None or args.input.isdigit():
        device_id = int(args.input) if args.input else 0
        source    = CameraSource(device_id)
    else:
        source = VideoFileSource(args.input)

    run_pipeline(
        source    = source,
        mode      = args.mode,
        test_mode = not args.prod,
        show      = not args.no_show,
        save_path = args.output,
    )


if __name__ == "__main__":
    main()
