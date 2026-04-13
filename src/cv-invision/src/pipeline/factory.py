import queue
from src.pipeline.const import (
    QUEUE_SIZE_READER, QUEUE_SIZE_DETECTOR, QUEUE_SIZE_TRACKER,
    READER_DROP_POLICY, DETECTOR_DROP_POLICY, TRACKER_DROP_POLICY,
    THREAD_READER, THREAD_DETECTOR, THREAD_TRACKER, THREAD_RENDERER,
    DropPolicy
)
from src.pipeline.sources import VideoFileSource
from src.pipeline.stages.reader   import ReaderStage
from src.pipeline.stages.detector import DetectorStage
from src.pipeline.stages.tracker  import TrackerStage
from src.pipeline.stages.renderer import RendererStage
from src.utils.log import get_logger

logger = get_logger(__name__)


def build_tracking_pipeline(source, model, tracker): # ← Removed 'output' arg
    logger.info("🏭 Building tracking pipeline...")

    # 1. Determine policy based on source type
    if isinstance(source, VideoFileSource):
        reader_policy = DropPolicy.BLOCK
    else:
        reader_policy = DropPolicy.DROP_OLDEST

    # 2. Setup queues (Added output_queue)
    frame_queue     = queue.Queue(maxsize=QUEUE_SIZE_READER)
    detection_queue = queue.Queue(maxsize=QUEUE_SIZE_DETECTOR)
    tracks_queue    = queue.Queue(maxsize=QUEUE_SIZE_TRACKER)
    output_queue    = queue.Queue(maxsize=10) # ← Catch-all for main thread

    # 3. Configure stages
    stages = [
        ReaderStage(
            name        = THREAD_READER,
            source      = source,
            out_queue   = frame_queue,
            drop_policy = reader_policy, # ← Using dynamic policy
        ),
        DetectorStage(
            name        = THREAD_DETECTOR,
            model       = model,
            in_queue    = frame_queue,
            out_queue   = detection_queue,
            drop_policy = DETECTOR_DROP_POLICY,
        ),
        TrackerStage(
            name        = THREAD_TRACKER,
            tracker     = tracker,
            in_queue    = detection_queue,
            out_queue   = tracks_queue,
            drop_policy = TRACKER_DROP_POLICY,
        ),
        RendererStage(
            name      = THREAD_RENDERER,
            in_queue  = tracks_queue,
            out_queue = output_queue, # ← Now points to output_queue
            # Removed 'output = output' line
        ),
    ]

    logger.info(f"✅ Pipeline built — {len(stages)} stages ready")
    return stages, output_queue # ← Returning both now


def build_detection_pipeline(source, model): # ← Removed 'output' arg
    logger.info("🏭 Building detection-only pipeline...")

    if isinstance(source, VideoFileSource):
        reader_policy = DropPolicy.BLOCK
    else:
        reader_policy = DropPolicy.DROP_OLDEST

    frame_queue     = queue.Queue(maxsize=QUEUE_SIZE_READER)
    detection_queue = queue.Queue(maxsize=QUEUE_SIZE_DETECTOR)
    output_queue    = queue.Queue(maxsize=10) # ← Added

    stages = [
        ReaderStage(
            name        = THREAD_READER,
            source      = source,
            out_queue   = frame_queue,
            drop_policy = reader_policy,
        ),
        DetectorStage(
            name        = THREAD_DETECTOR,
            model       = model,
            in_queue    = frame_queue,
            out_queue   = detection_queue,
            drop_policy = DETECTOR_DROP_POLICY,
        ),
        RendererStage(
            name      = THREAD_RENDERER,
            in_queue  = detection_queue,
            out_queue = output_queue, # ← Added
        ),
    ]

    logger.info(f"✅ Pipeline built — {len(stages)} stages ready")
    return stages, output_queue # ← Returning both
