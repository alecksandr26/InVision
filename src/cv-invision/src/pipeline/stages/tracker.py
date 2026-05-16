from src.pipeline.stages.base import BaseStage
from src.utils.log            import get_logger

logger = get_logger(__name__)

class TrackerStage(BaseStage):
    def __init__(self, name, tracker, in_queue, out_queue, drop_policy):
        super().__init__(
            name        = name,
            in_queue    = in_queue,
            out_queue   = out_queue,
            drop_policy = drop_policy,
        )
        self.tracker = tracker

    def process(self, data):
        try:
            ds_detections = [
                (d[0], d[1], d[2])   # (xywh, conf, class_id)
                for d in data.detections
            ]
            
            # Pass original raw bbox to supplementary
            supplementary_bboxes = [d[3] for d in data.detections] 

            tracks = self.tracker.update_tracks(
                ds_detections,
                frame  = data.frame,
                others = supplementary_bboxes
            )

            data.tracks = [t for t in tracks if t.is_confirmed()]

            logger.debug(f"  [tracker] frame {data.frame_idx} — {len(data.detections)} detections → {len(data.tracks)} confirmed tracks")

        except Exception as e:
            logger.error(f"❌ Tracker error on frame {data.frame_idx}: {e}")
            data.tracks = []

        return data
