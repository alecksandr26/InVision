from deep_sort_realtime.deepsort_tracker import DeepSort
from .const import TRACKER_MAX_AGE, TRACKER_N_INIT, TRACKER_NMS_MAX_OVERLAP, \
    TRACKER_MAX_COSINE_DISTANCE, TRACKER_NN_BUDGET, TRACKER_HALF, TRACKER_BGR, \
    TRACKER_EMBEDDER



# ============================================================
# TRACKER FACTORY
# ============================================================

def build_tracker(test_mode: bool = False) -> DeepSort:
    """
    Builds and returns a configured DeepSort tracker instance.

    Args:
        test_mode: If True uses n_init=1 for instant track confirmation —
                   useful for quick testing. If False uses TRACKER_N_INIT (production).

    Returns:
        DeepSort: Configured tracker instance ready for inference loop.

    Usage:
        from src.model.tracker import build_tracker

        # testing
        tracker = build_tracker(test_mode=True)

        # production
        tracker = build_tracker(test_mode=False)

        # in inference loop:
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
    """
    n_init = 1 if test_mode else TRACKER_N_INIT

    tracker = DeepSort(
        max_age             = TRACKER_MAX_AGE,
        n_init              = n_init,
        nms_max_overlap     = TRACKER_NMS_MAX_OVERLAP,
        max_cosine_distance = TRACKER_MAX_COSINE_DISTANCE,
        nn_budget           = TRACKER_NN_BUDGET,
        embedder            = TRACKER_EMBEDDER,   # we pass our own 256-dim embeddings
        half                = TRACKER_HALF,
        bgr                 = TRACKER_BGR,
    )

    return tracker
