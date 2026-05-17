from cv_invision.pipeline.stages.base    import BaseStage
from cv_invision.pipeline.sources        import BaseSource
from cv_invision.utils.log               import get_logger
import queue
import time

logger = get_logger(__name__)


class FrameData:
    """
    Data structure passed between pipeline stages.
    Each stage adds its own results and passes it downstream.
    """
    def __init__(self, frame, frame_idx: int, timestamp: float):
        self.frame       = frame        # original BGR frame
        self.frame_idx   = frame_idx    # frame number
        self.timestamp   = timestamp    # time.time() when frame was read
        self.detections  = []           # filled by DetectorStage
        self.tracks      = []           # filled by TrackerStage
        self.annotated   = None         # filled by RendererStage


class ReaderStage(BaseStage):
    """
    Stage 0 — Frame Reader (Core 0)
    Reads frames from a source and pushes them into the frame queue.

    Input  : BaseSource (VideoFileSource, CameraSource, StreamSource)
    Output : FrameData → frame_queue
    """

    def __init__(self, name, source: BaseSource, out_queue, drop_policy):
        super().__init__(
            name        = name,
            in_queue    = None,      # ← no input queue, reads from source
            out_queue   = out_queue,
            drop_policy = drop_policy,
        )
        self.source    = source
        self.frame_idx = 0

    def _run(self):
        """Override _run since reader has no input queue — reads from source directly."""
        logger.info(f"✅ Stage '{self.name}' running...")

        while not self._stop_event.is_set():
            ret, frame = self.source.read()

            if not ret:
                logger.info(f"📽️  Source exhausted — sending sentinel")
                self._put(None)  # signal downstream to stop
                break

            self.frame_idx += 1

            data = FrameData(
                frame     = frame,
                frame_idx = self.frame_idx,
                timestamp = time.time(),
            )

            if data is not None:
                for callback in self._callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"❌ Error in stage [{self.name}] callback: {e}")
                self._put(data)

        self.source.release()
        logger.info(f"✅ Stage '{self.name}' done — {self.frame_idx} frames read")

    def process(self, item):
        pass  # not used — reader overrides _run directly
