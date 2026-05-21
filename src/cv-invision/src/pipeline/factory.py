import queue
from cv_invision.pipeline.const import (
    QUEUE_SIZE_READER, QUEUE_SIZE_DETECTOR, QUEUE_SIZE_TRACKER,
    READER_DROP_POLICY, DETECTOR_DROP_POLICY, TRACKER_DROP_POLICY,
    THREAD_READER, THREAD_DETECTOR, THREAD_TRACKER, THREAD_RENDERER,
    DropPolicy
)
from cv_invision.pipeline.sources import VideoFileSource
from cv_invision.pipeline.stages.reader   import ReaderStage
from cv_invision.pipeline.stages.detector import DetectorStage
from cv_invision.pipeline.stages.tracker  import TrackerStage
from cv_invision.pipeline.stages.renderer import RendererStage
from cv_invision.utils.log import get_logger

logger = get_logger(__name__)

def build_full_pipeline(source, model, tracker):
    logger.info("🏭 Building full pipeline...")

    if isinstance(source, VideoFileSource):
        reader_policy   = DropPolicy.BLOCK
        renderer_policy = DropPolicy.BLOCK
    else:
        reader_policy   = DropPolicy.DROP_OLDEST
        renderer_policy = DropPolicy.DROP_OLDEST

    frame_queue     = queue.Queue(maxsize=QUEUE_SIZE_READER)
    detection_queue = queue.Queue(maxsize=QUEUE_SIZE_DETECTOR)
    tracks_queue    = queue.Queue(maxsize=QUEUE_SIZE_TRACKER)
    output_queue    = queue.Queue(maxsize=10)

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
        TrackerStage(
            name        = THREAD_TRACKER,
            tracker     = tracker,
            in_queue    = detection_queue,
            out_queue   = tracks_queue,
            drop_policy = TRACKER_DROP_POLICY,
        ),
        RendererStage(
            name        = THREAD_RENDERER,
            in_queue    = tracks_queue,
            out_queue   = output_queue,
            drop_policy = renderer_policy, 
        ),
    ]

    logger.info(f"✅ Pipeline built — {len(stages)} stages ready")
    return stages, output_queue

def build_tracking_pipeline(source, model, tracker):
    return build_full_pipeline(source, model, tracker)

def build_detection_pipeline(source, model):
    logger.info("🏭 Building detection-only pipeline...")

    if isinstance(source, VideoFileSource):
        reader_policy   = DropPolicy.BLOCK
        renderer_policy = DropPolicy.BLOCK
    else:
        reader_policy   = DropPolicy.DROP_OLDEST
        renderer_policy = DropPolicy.DROP_OLDEST

    frame_queue     = queue.Queue(maxsize=QUEUE_SIZE_READER)
    detection_queue = queue.Queue(maxsize=QUEUE_SIZE_DETECTOR)
    output_queue    = queue.Queue(maxsize=10) 

    stages = [
        ReaderStage(name=THREAD_READER, source=source, out_queue=frame_queue, drop_policy=reader_policy),
        DetectorStage(name=THREAD_DETECTOR, model=model, in_queue=frame_queue, out_queue=detection_queue, drop_policy=DETECTOR_DROP_POLICY),
        RendererStage(name=THREAD_RENDERER, in_queue=detection_queue, out_queue=output_queue, drop_policy=renderer_policy),
    ]
    return stages, output_queue

def build_headless_pipeline(source, model, tracker):
    """
    🏭 Builds a high-performance headless tracking pipeline without RendererStage.
    The final output queue connects directly to the TrackerStage output buffer.
    """
    logger.info("⚡ Building Headless Pipeline (No UI/Renderer)...")

    if isinstance(source, VideoFileSource):
        reader_policy = DropPolicy.BLOCK
    else:
        reader_policy = DropPolicy.DROP_OLDEST

    # Instantiate queues up to the Tracker stage
    frame_queue     = queue.Queue(maxsize=QUEUE_SIZE_READER)
    detection_queue = queue.Queue(maxsize=QUEUE_SIZE_DETECTOR)
    output_queue    = queue.Queue(maxsize=QUEUE_SIZE_TRACKER)  # Straight from tracker to your loop

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
        TrackerStage(
            name        = THREAD_TRACKER,
            tracker     = tracker,
            in_queue    = detection_queue,
            out_queue   = output_queue,
            drop_policy = TRACKER_DROP_POLICY,
        ),
    ]

    logger.info(f"✅ Headless Pipeline built — {len(stages)} stages ready (Renderer Bypassed)")
    return stages, output_queue
