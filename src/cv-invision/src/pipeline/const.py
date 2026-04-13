from pathlib import Path

# ============================================================
# PIPELINE STAGE CONFIG
# ============================================================

# Maximum frames buffered between stages.
# Too large  → high latency (old frames pile up)
# Too small  → stages block each other
QUEUE_SIZE_READER   = 2   # reader   → detector
QUEUE_SIZE_DETECTOR = 2   # detector → tracker
QUEUE_SIZE_TRACKER  = 2   # tracker  → renderer

# ============================================================
# FRAME DROP POLICY
# ============================================================
# What to do when a queue is full:
#   DROP_OLDEST → remove oldest frame, add new one (lowest latency)
#   DROP_NEWEST → discard incoming frame (preserves older context)
#   BLOCK       → wait until space available (no drops, higher latency)

class DropPolicy:
    DROP_OLDEST = "drop_oldest"  # ← best for real-time video/stream
    DROP_NEWEST = "drop_newest"  # ← best for recording/saving
    BLOCK       = "block"        # ← best for offline video processing

# Default policy per stage
READER_DROP_POLICY   = DropPolicy.DROP_OLDEST  # real-time priority
DETECTOR_DROP_POLICY = DropPolicy.DROP_OLDEST
TRACKER_DROP_POLICY  = DropPolicy.DROP_OLDEST

# ============================================================
# THREAD CONFIG
# ============================================================

# Thread names — useful for debugging and logs
THREAD_READER   = "invision.reader"
THREAD_DETECTOR = "invision.detector"
THREAD_TRACKER  = "invision.tracker"
THREAD_RENDERER = "invision.renderer"

# Timeout in seconds for queue get/put operations
# Prevents threads from blocking forever if pipeline stops
QUEUE_TIMEOUT = 1.0

# ============================================================
# PIPELINE SENTINEL
# ============================================================

# Special value pushed to queues to signal shutdown
# Each stage passes it downstream when it sees it
SENTINEL = None
