"""
InVision - Real-Time Camera Detection Test
Tests the full pipeline: Camera → Detection Model → Tracker → Display
"""

import cv2
import numpy as np
import sys
import time
import logging
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("test_cam")

# ── Colors BGR (OpenCV native) ────────────────────────────────────────────────
COLORS = [
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 128, 255),  # orange
    (128, 0, 255),  # purple
]

def draw_detection(frame, x1, y1, x2, y2, label, conf, track_id=None, color=(0, 255, 0)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag  = f"#{track_id} " if track_id is not None else ""
    text = f"{tag}{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

def draw_hud(frame, fps, det_count, track_count, inf_ms):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 95), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "InVision - Real Time Detection", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS     : {fps:5.1f}", (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Infer   : {inf_ms:5.1f} ms", (8, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Det/Trk : {det_count} / {track_count}", (8, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    logger.info("=" * 55)
    logger.info("  InVision Camera Test — Starting up")
    logger.info("=" * 55)

    # ── 1. Load detection model ────────────────────────────────────────────────
    logger.info("Loading production detection model...")
    from ..model.detection_model import load_detection_model_prod
    model = load_detection_model_prod()
    logger.info("Detection model ready ✅")

    # ── 2. Build tracker ───────────────────────────────────────────────────────
    logger.info("Building tracker...")
    from ..model.tracker import build_tracker
    tracker = build_tracker(test_mode=False)
    logger.info("Tracker ready ✅")

    # ── 3. Open camera ─────────────────────────────────────────────────────────
    logger.info("Initializing camera...")
    use_picamera = False
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        # ⚡ Optimization: capture at 640x640 directly — no resize needed before inference
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 640)},  # BGR directly, no cvtColor needed
            raw={"size": (1640, 1232)},                     # smaller raw = faster capture
            controls={"FrameRate": 30},
        )
        picam2.configure(config)
        picam2.start()
        use_picamera = True
        logger.info("PiCamera2 started ✅")
    except Exception as e:
        logger.warning(f"PiCamera2 failed ({e}), falling back to cv2.VideoCapture")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        if not cap.isOpened():
            logger.error("No camera found. Exiting.")
            sys.exit(1)
        logger.info("OpenCV VideoCapture started ✅")

    # ── 4. Inference loop ──────────────────────────────────────────────────────
    logger.info("Starting inference loop — press Q to quit")
    fps         = 0.0
    inf_ms      = 0.0
    frame_count = 0
    t_start     = time.time()

    try:
        while True:
            t0 = time.time()

            # ── Grab frame ────────────────────────────────────────────────────
            if use_picamera:
                # ⚡ BGR888 config means NO color conversion needed
                frame = picam2.capture_array()
            else:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera read failed, retrying...")
                    continue

            # ── Detection ─────────────────────────────────────────────────────
            t_inf = time.time()
            results   = model(frame, verbose=False)
            inf_ms    = (time.time() - t_inf) * 1000  # ms

            result    = results[0]
            boxes     = result.boxes
            det_count = len(boxes)

            # ── DeepSort input ────────────────────────────────────────────────
            ds_input = []
            for i in range(det_count):
                xyxy  = np.array(boxes.xyxy[i]).astype(int)
                conf  = float(np.array(boxes.conf[i]))
                cls   = int(np.array(boxes.cls[i]))
                label = result.names.get(cls, str(cls))
                x1, y1, x2, y2 = xyxy
                ds_input.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

            # ── Tracker update ────────────────────────────────────────────────
            tracks      = tracker.update_tracks(ds_input, frame=frame)
            track_count = 0

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_count += 1
                track_id    = int(track.track_id)
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                label = track.get_det_class() or "object"
                conf  = track.get_det_conf() or 0.0
                color = COLORS[track_id % len(COLORS)]
                draw_detection(frame, x1, y1, x2, y2, label, conf, track_id, color)

            # ── HUD + display ─────────────────────────────────────────────────
            draw_hud(frame, fps, det_count, track_count, inf_ms)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # fix colors
            cv2.imshow("InVision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Q pressed — shutting down")
                break

            # ── FPS ───────────────────────────────────────────────────────────
            frame_count += 1
            fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - t0, 1e-6))

            if frame_count % 30 == 0:
                logger.info(f"Frame {frame_count:5d} | FPS: {fps:.1f} | "
                            f"Inference: {inf_ms:.1f}ms | "
                            f"Det: {det_count} | Tracks: {track_count} | "
                            f"Uptime: {time.time() - t_start:.0f}s")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Cleaning up...")
        if use_picamera:
            picam2.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("Done 👋")

if __name__ == "__main__":
    main()
