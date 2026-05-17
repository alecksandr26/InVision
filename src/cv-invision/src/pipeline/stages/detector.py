import numpy as np
from cv_invision.pipeline.stages.base import BaseStage
from cv_invision.model.const          import TRACKER_MIN_CONF
from cv_invision.utils.log            import get_logger

logger = get_logger(__name__)


class DetectorStage(BaseStage):
    """
    Stage 1 — YOLO Detector (Core 1)
    Runs YOLO inference on each frame and extracts raw detections.

    Input  : FrameData (frame)          ← from frame_queue
    Output : FrameData (+ detections)   → detection_queue

    detections format: [([x, y, w, h], conf, class_id), ...]
    """

    def __init__(self, name, model, in_queue, out_queue, drop_policy):
        super().__init__(
            name        = name,
            in_queue    = in_queue,
            out_queue   = out_queue,
            drop_policy = drop_policy,
        )
        self.model = model

    def process(self, data):
        """
        Runs YOLO inference on the frame and attaches
        formatted detections to the FrameData object.

        Args:
            data: FrameData from ReaderStage

        Returns:
            FrameData with detections filled in
        """
        try:
            results = self.model(data.frame)

            detections = []
            for result in results:
                for box in result.boxes:
                    conf     = float(box.conf[0])
                    if conf < TRACKER_MIN_CONF:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id        = int(box.cls[0])
                    detections.append((
                        [x1, y1, x2 - x1, y2 - y1],  # xywh
                        conf,
                        class_id,
                        [x1, y1, x2, y2],             # original xyxy
                    ))

            data.detections = detections
            logger.debug(f"  [detector] frame {data.frame_idx} — {len(detections)} detection(s)")

        except Exception as e:
            logger.error(f"❌ Detector error on frame {data.frame_idx}: {e}")
            data.detections = []

        return data
