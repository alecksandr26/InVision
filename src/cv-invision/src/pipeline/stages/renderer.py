import cv2
import numpy as np
from src.pipeline.stages.base import BaseStage
from src.model.const          import CLASS_NAMES
from src.utils.log            import get_logger

logger = get_logger(__name__)

COLORS = [
    (255,  56,  56), ( 56, 152, 255), (255, 157,  56), (100, 100, 255),
    ( 56, 255, 122), (255, 255,  56), ( 56, 255, 255), (255,  56, 193),
    (173, 255,  47), (255, 128,   0),
]

def get_color(class_id: int) -> tuple:
    return COLORS[class_id % len(COLORS)]

class RendererStage(BaseStage):
    def __init__(self, name, in_queue, out_queue, drop_policy):
        super().__init__(
            name        = name,
            in_queue    = in_queue,
            out_queue   = out_queue,
            drop_policy = drop_policy,  # Dynamic injection fixed
        )

    def _render_tracks(self, frame, tracks):
        confirmed_count = 0
        for track in tracks:
            # Successfully accesses YOLO's raw box now via get_det_supplementary
            bbox = track.get_det_supplementary() 
            if bbox is None:
                bbox = track.to_ltrb()
                
            x1, y1, x2, y2 = map(int, bbox)
            track_id = track.track_id
            class_id = track.det_class

            color = get_color(class_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            confirmed_count += 1

        return frame, confirmed_count

    def _render_hud(self, frame, frame_idx, confirmed):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(
            frame,
            f"Frame: {frame_idx}  |  Tracks: {confirmed}  |  Q to quit",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        return frame

    def process(self, data):
        try:
            frame = data.frame.copy()
            frame, confirmed = self._render_tracks(frame, data.tracks)
            frame = self._render_hud(frame, data.frame_idx, confirmed)
            data.annotated = frame
            
            logger.debug(f"  [renderer] frame {data.frame_idx} — {confirmed} track(s)")
        except Exception as e:
            logger.error(f"❌ Renderer error on frame {data.frame_idx}: {e}")
            
        return data
