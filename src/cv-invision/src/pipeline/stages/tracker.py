from src.pipeline.stages.base import BaseStage
from src.utils.log            import get_logger

logger = get_logger(__name__)


class TrackerStage(BaseStage):
    """
    Stage 2 — DeepSort Tracker (Core 2)
    Updates tracker with new detections and returns confirmed tracks.

    Input  : FrameData (+ detections)  ← from detection_queue
    Output : FrameData (+ tracks)      → tracks_queue

    tracks: list of confirmed DeepSort track objects
    """

    def __init__(self, name, tracker, in_queue, out_queue, drop_policy):
        super().__init__(
            name        = name,
            in_queue    = in_queue,
            out_queue   = out_queue,
            drop_policy = drop_policy,
        )
        self.tracker = tracker

    def process(self, data):
        """
        Updates DeepSort tracker with detections from DetectorStage.
        Attaches confirmed tracks to FrameData.

        Args:
            data: FrameData with detections from DetectorStage

        Returns:
            FrameData with tracks filled in
        """
        try:
            # format detections for DeepSort — strip supplementary xyxy
            ds_detections = [
                (d[0], d[1], d[2])   # (xywh, conf, class_id)
                for d in data.detections
            ]

            tracks = self.tracker.update_tracks(
                ds_detections,
                frame = data.frame
            )

            # keep only confirmed tracks
            data.tracks = [t for t in tracks if t.is_confirmed()]

            logger.debug(
                f"  [tracker] frame {data.frame_idx} — "
                f"{len(data.detections)} detections → "
                f"{len(data.tracks)} confirmed tracks"
            )

        except Exception as e:
            logger.error(f"❌ Tracker error on frame {data.frame_idx}: {e}")
            data.tracks = []

        return data
