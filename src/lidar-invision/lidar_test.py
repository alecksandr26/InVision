"""
lidar_test.py — validates the install and shows the exact usage pattern
for syncing LiDAR scans with camera frames in an AI inference loop.

Run:
    python lidar_test.py                  # uses auto-detected port
    python lidar_test.py /dev/ttyUSB0     # explicit port
"""

import sys
import time
import numpy as np

# ── Import ────────────────────────────────────────────────────────────────────
try:
    from lidar_invision import LidarManager, Config
except ImportError as e:
    print(f"❌ Could not import lidar_invision: {e}")
    print("   Run:  pip install .  or rebuilding the module wrapper workspace.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Basic smoke test (Checks if bindings load and configuration registers)
# ─────────────────────────────────────────────────────────────────────────────
def test_import():
    mgr = LidarManager()
    print(f"✓ LidarManager created: {mgr}")

    cfg = Config()
    cfg.port        = ""  # Default empty for robust system-level auto detection
    cfg.baudrate    = 230400
    cfg.frequency   = 10.0
    cfg.min_angle   = -180.0
    cfg.max_angle   =  180.0
    cfg.min_dist_cm = 5.0
    cfg.max_dist_cm = 6400.0  # Safe max clearance boundary matching hardware specs

    mgr.configure(cfg)
    print(f"✓ Config structure successfully created and applied.")
    print(f"  is_running = {mgr.is_running}")


# ─────────────────────────────────────────────────────────────────────────────
#  2. Full hardware test (Polls active stream matrices from the Tmini)
# ─────────────────────────────────────────────────────────────────────────────
def test_hardware(port=None):
    print("\n── Hardware test ────────────────────────────────────────────────")
    mgr = LidarManager()

    cfg = Config()
    if port:
        cfg.port = port
    else:
        cfg.port = ""  # Let C++ framework handle link scanning automatically
        
    cfg.baudrate    = 230400  # Crucial hardware match for your Tmini Plus
    cfg.frequency   = 10.0
    cfg.min_angle   = -180.0
    cfg.max_angle   =  180.0
    cfg.min_dist_cm = 5.0
    cfg.max_dist_cm = 6400.0

    print(f"Connecting to YDLIDAR... (Auto-detect active if empty: '{cfg.port}')")
    mgr.configure(cfg)

    print("Starting background telemetry loop...")
    mgr.start()

    # Accessing read-only property safely without parentheses
    if mgr.has_error:
        print(f"❌ Startup error: {mgr.last_error}")
        return

    print(f"✓ Running: {mgr}")

    # ── Wait for the first scan to arrive ────────────────────────────────
    print("Waiting for first scan array to clear processing queues…")
    deadline = time.monotonic() + 7.0
    while time.monotonic() < deadline:
        scan = mgr.fetch_scan(wait_for_fresh=False, timeout_ms=10)
        if scan.scan_index > 0:
            break
        time.sleep(0.05)
    else:
        print("❌ No scan frames cleared queues. Check hardware connection.")
        mgr.stop()
        return

    # ── Show a few scans ──────────────────────────────────────────────────
    print(f"\n{'Index':>6}  {'Points':>6}  {'Age ms':>8}  "
          f"{'Min dist cm':>12}  {'Max dist cm':>12}")
    print("─" * 55)

    for _ in range(10):
        scan = mgr.fetch_scan(wait_for_fresh=False, timeout_ms=0)
        pts  = scan.points                 # Extracted array structure: shape (N, 3)

        if pts.shape[0] > 0:
            dist_col   = pts[:, 1]         # Column Index 1 tracks distance measurements
            min_d      = float(dist_col.min())
            max_d      = float(dist_col.max())
        else:
            min_d = max_d = 0.0

        print(f"{scan.scan_index:>6}  {pts.shape[0]:>6}  "
              f"{scan.age_ms:>8.1f}  {min_d:>12.1f}  {max_d:>12.1f}")
        
        time.sleep(0.1)

    # ── Demonstrate wait_for_fresh ────────────────────────────────────────
    print("\n── wait_for_fresh demo (blocks until next scan arrives) ─────────")
    t0   = time.perf_counter()
    scan = mgr.fetch_scan(wait_for_fresh=True, timeout_ms=200)
    dt   = (time.perf_counter() - t0) * 1000
    print(f"   Got scan #{scan.scan_index} in {dt:.1f} ms  "
          f"(age {scan.age_ms:.1f} ms,  {scan.points.shape[0]} pts)")

    # ── Simulated AI inference loop ───────────────────────────────────────
    print("\n── Simulated AI inference loop (10 iterations) ──────────────────")
    print("   Pattern: grab_frame() → fetch_scan() → [inference 100 ms] → fuse\n")

    for i in range(10):
        t_frame = time.perf_counter()

        # Zero-overhead snapshot fetch synced to the target camera frame capture instant
        scan = mgr.fetch_scan(wait_for_fresh=False, timeout_ms=0)

        t_fetch = time.perf_counter()
        fetch_us = (t_fetch - t_frame) * 1e6

        # Simulating frame processing latency overhead
        time.sleep(0.1)

        pts        = scan.points
        n_close    = int((pts[:, 1] < 50).sum()) if pts.shape[0] > 0 else 0

        print(f"   [{i:02d}] fetch: {fetch_us:5.0f} µs  "
              f"scan_age: {scan.age_ms:5.1f} ms  "
              f"pts: {pts.shape[0]:4d}  "
              f"close(<50cm): {n_close}")

    print("\n── Stopping background processes ─────────────────────────────────")
    mgr.stop()
    print(f"✓ Stopped. Total scans processed: {mgr.scan_count}")
    print("   LiDAR testing pipeline executed successfully.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_import()

    port = sys.argv[1] if len(sys.argv) > 1 else ""

    try:
        test_hardware(port)
    except KeyboardInterrupt:
        print("\nInterrupted by operator signal layout.")
