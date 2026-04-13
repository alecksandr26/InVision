import cv2
import numpy as np
from src.pipeline.stages.base import BaseStage
from src.model.const          import CLASS_NAMES
from src.utils.log            import get_logger

logger = get_logger(__name__)

# ── colors per class ──────────────────────────────────────────
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
    return COLORS[class_id % len(COLORS)]


class RendererStage(BaseStage):
    """
    Stage 3 — Renderer + Output (Core 3)
    Renders bboxes, track IDs, HUD overlay on the frame
    and sends to output handler (display, save, stream).

    Input  : FrameData (+ tracks)  ← from tracks_queue
    Output : None (end of pipeline)
    """

    def __init__(self, name, in_queue, out_queue=None, output=None):
        super().__init__(
            name        = name,
            in_queue    = in_queue,
            out_queue   = out_queue,
            drop_policy = "drop_oldest",
        )
        self.output      = output
        self.window_name = "InVision Detection"

    def _render_tracks(self, frame: np.ndarray, tracks: list) -> tuple:
        """Renders confirmed tracks on frame."""
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

            # use original YOLO bbox if available
            supplementary   = track.get_det_supplementary()
            if supplementary is not None:
                x1, y1, x2, y2 = map(int, supplementary)
            else:
                x1, y1, x2, y2 = map(int, track.to_ltrb())

            # bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

            # label background
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - baseline - 4), (x1 + lw, y1), color, -1)

            # label text
            cv2.putText(
                frame, label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.5,
                color     = (0, 0, 0),
                thickness = 1,
                lineType  = cv2.LINE_AA
            )

        return frame, confirmed

    def _render_hud(self, frame, frame_idx, confirmed):
        """Renders HUD overlay."""
        h, w    = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(
            frame,
            f"Frame: {frame_idx}  |  Tracks: {confirmed}  |  Q to quit",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.55,
            color     = (255, 255, 255),
            thickness = 1,
            lineType  = cv2.LINE_AA
        )
        return frame

    def process(self, data):
        """
        Renders tracks and HUD on frame then
        sends to output handler.

        Args:
            data: FrameData with tracks from TrackerStage

        Returns:
            None — end of pipeline
        """
        try:
            frame = data.frame.copy()

            # render tracks
            frame, confirmed = self._render_tracks(frame, data.tracks)

            # render HUD
            frame = self._render_hud(frame, data.frame_idx, confirmed)

            data.annotated = frame

            
            logger.debug(
                f"  [renderer] frame {data.frame_idx} — "
                f"{confirmed} track(s) rendered"
            )

        except Exception as e:
            logger.error(f"❌ Renderer error on frame {data.frame_idx}: {e}")
            return None  # ← end of pipeline, nothing downstream
        return data
