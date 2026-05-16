from pathlib import Path

# ============================================================
# PIPELINE STAGE CONFIG
# ============================================================

# Increased queues to absorb YOLO latency spikes on the Raspberry Pi
QUEUE_SIZE_READER   = 8   # reader   → detector
QUEUE_SIZE_DETECTOR = 4   # detector → tracker
QUEUE_SIZE_TRACKER  = 4   # tracker  → renderer

# ============================================================
# FRAME DROP POLICY
# ============================================================

class DropPolicy:
    DROP_OLDEST = "drop_oldest"  # best for real-time video/stream
    DROP_NEWEST = "drop_newest"  # best for recording/saving
    BLOCK       = "block"        # best for offline video processing

# Default policy per stage
READER_DROP_POLICY   = DropPolicy.DROP_OLDEST  
DETECTOR_DROP_POLICY = DropPolicy.DROP_OLDEST
TRACKER_DROP_POLICY  = DropPolicy.DROP_OLDEST

# ============================================================
# THREAD CONFIG
# ============================================================

THREAD_READER   = "invision.reader"
THREAD_DETECTOR = "invision.detector"
THREAD_TRACKER  = "invision.tracker"
THREAD_RENDERER = "invision.renderer"

QUEUE_TIMEOUT = 1.0

# Special value pushed to queues to signal shutdown
SENTINEL = None
