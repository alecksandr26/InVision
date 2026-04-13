import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import argparse
import numpy as np
from pathlib import Path

from ..detection_model  import load_detection_model_test, load_detection_model_prod
from ..tracker import build_tracker
from ..const            import CLASS_NAMES, TRACKER_MIN_CONF
from src.utils.const    import VIDEO_EXTENSIONS
from src.utils.log      import get_logger

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

def render_tracks(frame: np.ndarray, tracks: list) -> tuple[np.ndarray, int]:
    """
    Renders confirmed DeepSort tracks as bounding boxes with track IDs on a frame.

    Args:
        frame : BGR frame as numpy array
        tracks: List of DeepSort track objects from tracker.update_tracks()

    Returns:
        tuple:
            np.ndarray : Annotated frame with bboxes, track IDs and class labels
            int        : Number of confirmed tracks rendered
    """
    confirmed = 0

    for track in tracks:
        if not track.is_confirmed():
            continue

        confirmed      += 1
        track_id        = track.track_id
        class_id        = int(track.get_det_class()) if track.get_det_class() is not None else 0
        class_name      = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
        color           = get_color(class_id)
        label           = f"[{track_id}] {class_name}"

        # ← use original YOLO detection bbox instead of Kalman prediction
        det = track.get_det_supplementary()
        if det is not None:
            x1, y1, x2, y2 = map(int, det)
        else:
            # fallback to Kalman prediction if no fresh detection this frame
            x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Draw bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - baseline - 4),
            (x1 + label_w, y1),
            color,
            thickness=-1
        )

        # Draw label text
        cv2.putText(
            frame, label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.5,
            color     = (0, 0, 0),
            thickness = 1,
            lineType  = cv2.LINE_AA
        )

        logger.debug(f"  [Track {track_id}] {class_name} bbox=({x1},{y1},{x2},{y2})")

    return frame, confirmed



def render_hud(frame: np.ndarray, frame_idx: int, total_frames: int, fps: float, detections: int) -> np.ndarray:
    """
    Renders a HUD overlay on the frame with playback and detection info.

    Args:
        frame       : BGR frame as numpy array
        frame_idx   : Current frame index
        total_frames: Total number of frames in the video
        fps         : Video FPS
        detections  : Number of detections in current frame

    Returns:
        np.ndarray: Frame with HUD overlay
    """
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    hud_text = f"Frame: {frame_idx}/{total_frames}  |  FPS: {fps:.1f}  |  Detections: {detections}  |  Q to quit"
    cv2.putText(
        frame, hud_text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.55,
        color     = (255, 255, 255),
        thickness = 1,
        lineType  = cv2.LINE_AA
    )

    return frame


def run_video_inference(
    video_path  : Path,
    model,
    tracker,
    save        : bool = True,
    show        : bool = True,
    output_path : Path = None,
) -> Path:
    """
    Runs detection inference frame by frame on a video and renders bboxes.

    Args:
        video_path  : Path to the input video file
        model       : Loaded YOLO model instance
        save        : Whether to save the annotated video to disk
        show        : Whether to display the annotated video with cv2.imshow()
        output_path : Custom output path. Defaults to <input>_result.mp4

    Returns:
        Path to the saved output video (if save=True)
    """
    # Validate extension
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        logger.error(f"Unsupported video format: '{video_path.suffix}'")
        logger.error(f"Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        raise SystemExit(1)

    if not video_path.exists():
        logger.error(f"Video not found: '{video_path}'")
        raise FileNotFoundError(f"Video not found: '{video_path}'")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: '{video_path}'")
        raise RuntimeError(f"Could not open video: '{video_path}'")

    # Video properties
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"🎬 Video: '{video_path.name}'")
    logger.info(f"⚙️  Resolution  : {width}x{height}")
    logger.info(f"⚙️  FPS         : {fps:.1f}")
    logger.info(f"⚙️  Total frames: {total_frames}")

    # Setup video writer
    writer      = None
    output_path = output_path or video_path.parent / f"{video_path.stem}_result.mp4"

    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        logger.info(f"💾 Saving output to '{output_path}'")

    # Setup display window
    window_name = f"InVision Detection — {video_path.name}"
    if show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)
        logger.info("🖼️  Press 'Q' to stop...")

    # ── Frame loop ────────────────────────────────────────────
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run inference on frame
        results    = model(frame)
        detections = sum(len(r.boxes) for r in results)

        # Format detections for DeepSort → ([x1,y1,w,h], conf, class_id)
        ds_detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf     = float(box.conf[0])
                if conf < TRACKER_MIN_CONF:
                    continue
                class_id = int(box.cls[0])
                ds_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))


        # Update tracker
        tracks = tracker.update_tracks(ds_detections, frame=frame)

        # Render bboxes
        annotated, confirmed = render_tracks(frame.copy(), tracks)

        # Render HUD — use confirmed tracks count instead of raw detections
        annotated = render_hud(annotated, frame_idx, total_frames, fps, confirmed)


        # Save frame
        if save and writer:
            writer.write(annotated)

        # Show frame
        if show:
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("⏹️  Stopped by user.")
                break

        if frame_idx % 30 == 0:
            logger.info(f"⏱️  Frame {frame_idx}/{total_frames} — {detections} detection(s)")

    # ── Cleanup ───────────────────────────────────────────────
    cap.release()

    if writer:
        writer.release()
        logger.info(f"✅ Saved result to '{output_path}'")

    if show:
        cv2.destroyAllWindows()

    logger.info(f"✅ Done! Processed {frame_idx} frames.")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="🎬 InVision Detection Model — Video Inference",
        epilog=f"Supported formats: {', '.join(sorted(VIDEO_EXTENSIONS))}"
    )
    parser.add_argument(
        "--input", "-i",
        type     = str,
        required = True,
        help     = "Path to input video file"
    )
    parser.add_argument(
        "--output", "-o",
        type    = str,
        default = None,
        help    = "Path to save the annotated output video. Defaults to <input>_result.mp4"
    )
    parser.add_argument(
        "--no-save",
        action  = "store_true",
        default = False,
        help    = "Do not save the annotated output video"
    )
    parser.add_argument(
        "--no-show",
        action  = "store_true",
        default = False,
        help    = "Do not display the video window"
    )
    return parser.parse_args()


def main():
    args       = parse_args()
    video_path = Path(args.input)

    logger.info("🚀 Starting InVision video inference pipeline...")

    model = load_detection_model_test()
    tracker = build_tracker()

    run_video_inference(
        video_path  = video_path,
        model       = model,
        tracker     = tracker,
        save        = not args.no_save,
        show        = not args.no_show,
        output_path = Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()


