import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import argparse
import threading
import queue
from pathlib import Path

from src.pipeline.factory  import build_full_pipeline, build_detection_pipeline
from src.pipeline.sources  import VideoFileSource, CameraSource, PiCameraSource
from src.model.detection_model import load_detection_model_prod, load_detection_model_test
from src.model.tracker         import build_tracker
from src.utils.log             import get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="🚀 InVision Detection Pipeline")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to video file or camera device id")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save annotated output video")
    parser.add_argument("--mode", "-m", type=str, default="tracking", choices=["tracking", "detection", "full"])
    parser.add_argument("--picamera", action="store_true", default=False, help="Use PiCamera2 hardware abstraction")
    parser.add_argument("--prod", action="store_true", default=False, help="Use production model")
    parser.add_argument("--no-show", action="store_true", default=False, help="Do not display output window")
    return parser.parse_args()

def display_output(show: bool = True, save_path: str = None, fps: float = 20.0):
    writer     = None
    stop_event = threading.Event()

    def handler(data):
        nonlocal writer
        frame = data.annotated
        if frame is None:
            return

        if show:
            cv2.imshow("InVision Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()

        if save_path:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
            writer.write(frame)

    def cleanup():
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    return handler, stop_event, cleanup

def run_pipeline(source, mode, test_mode, show, save_path):
    model = load_detection_model_test() if test_mode else load_detection_model_prod()
    
    if mode in ["tracking", "full"]:
        tracker = build_tracker()
        stages, output_queue = build_full_pipeline(source, model, tracker)
    else:
        stages, output_queue = build_detection_pipeline(source, model)

    handler, ui_stop_event, cleanup = display_output(show=show, save_path=save_path, fps=source.fps)

    for stage in stages:
        stage.start()

    logger.info("🚀 Pipeline running... (Press 'Q' to quit)")
    
    try:
        while not ui_stop_event.is_set():
            try:
                # Short timeout so we can constantly verify threads are alive
                data = output_queue.get(timeout=0.1)
                
                if data is None: 
                    break 
                    
                handler(data)

            except queue.Empty:
                # Fail-safe: Break out if all stages have died but dropped the sentinel
                if not any(stage._thread.is_alive() for stage in stages):
                    logger.warning("Pipeline stages terminated unexpectedly. Exiting main thread.")
                    break

    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt detected.")
    finally:
        logger.info("🧹 Initiating pipeline shutdown...")
        for stage in stages:
            stage.stop()
        for stage in stages:
            stage.join()
        cleanup()
        logger.info("✅ Pipeline shutdown complete.")

def main():
    args = parse_args()

    if args.picamera:
        source = PiCameraSource()
    elif args.input is None or args.input.isdigit():
        device_id = int(args.input) if args.input else 0
        source    = CameraSource(device_id)
    else:
        source = VideoFileSource(args.input)

    run_pipeline(
        source    = source,
        mode      = args.mode,
        test_mode = not args.prod,
        show      = not args.no_show,
        save_path = args.output
    )

if __name__ == "__main__":
    main()
