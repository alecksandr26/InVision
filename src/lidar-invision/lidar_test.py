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
    print("   Run:  pip install .  inside the lidar_module directory.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Basic smoke test  (no hardware needed — just checks module loads)
# ─────────────────────────────────────────────────────────────────────────────
def test_import():
    mgr = LidarManager()
    print(f"✓ LidarManager created: {mgr}")

    cfg = Config()
    cfg.port       = "/dev/ttyUSB0"
    cfg.frequency  = 10.0
    cfg.min_angle  = -180.0
    cfg.max_angle  =  180.0
    cfg.min_dist_cm = 5.0
    cfg.max_dist_cm = 150.0

    mgr.configure(cfg)
    print(f"✓ Config applied: {mgr.config}")
    print(f"  is_running = {mgr.is_running}")


# ─────────────────────────────────────────────────────────────────────────────
#  2. Full hardware test  (requires physical LiDAR connected)
# ─────────────────────────────────────────────────────────────────────────────
def test_hardware(port: str = ""):
    print("\n── Hardware test ────────────────────────────────────────────────")

    mgr = LidarManager()

    # Configure via individual setters (also valid, mirrors the Config object)
    if port:
        mgr.set_port(port)
    mgr.set_frequency(10.0)
    mgr.set_angle_range(-180.0, 180.0)     # full 360°
    mgr.set_distance_range(5.0, 150.0)     # 5 cm … 1.5 m

    print("Starting LiDAR background thread…")
    mgr.start()

    if mgr.has_error:
        print(f"❌ Startup error: {mgr.last_error}")
        return

    print(f"✓ Running: {mgr}")

    # ── Wait for the first scan to arrive ────────────────────────────────
    print("Waiting for first scan (up to 3 s)…")
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        scan = mgr.fetch_scan()
        if scan.scan_index > 0:
            break
        time.sleep(0.05)
    else:
        print("❌ No scan received within 3 s. Check hardware/port.")
        mgr.stop()
        return

    # ── Show a few scans ──────────────────────────────────────────────────
    print(f"\n{'Index':>6}  {'Points':>6}  {'Age ms':>8}  "
          f"{'Min dist cm':>12}  {'Max dist cm':>12}")
    print("─" * 55)

    for _ in range(10):
        scan = mgr.fetch_scan()                    # instant — no blocking
        pts  = scan.points                         # numpy (N, 3)

        if pts.shape[0] > 0:
            dist_col   = pts[:, 1]                 # column 1 = distance_cm
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
    print(f"  Got scan #{scan.scan_index} in {dt:.1f} ms  "
          f"(age {scan.age_ms:.1f} ms,  {scan.points.shape[0]} pts)")

    # ── Simulated AI inference loop ───────────────────────────────────────
    print("\n── Simulated AI inference loop (10 iterations) ──────────────────")
    print("  Pattern: grab_frame() → fetch_scan() → [inference 100 ms] → fuse\n")

    for i in range(10):
        # ① Simulate grabbing a camera frame
        # frame = camera.read()          # ← your real call
        t_frame = time.perf_counter()

        # ② Fetch LiDAR snapshot — happens in the SAME instant as the frame.
        #    Cost: one mutex lock + numpy copy  (~10–50 µs).
        scan = mgr.fetch_scan()

        t_fetch = time.perf_counter()
        fetch_us = (t_fetch - t_frame) * 1e6

        # ③ Simulate 100 ms AI inference
        # detections = model.infer(frame)    # ← your real call
        time.sleep(0.1)

        # ④ Fuse detections with scan
        pts        = scan.points               # numpy (N, 3)
        n_close    = int((pts[:, 1] < 50).sum()) if pts.shape[0] > 0 else 0

        print(f"  [{i:02d}] fetch: {fetch_us:5.0f} µs  "
              f"scan_age: {scan.age_ms:5.1f} ms  "
              f"pts: {pts.shape[0]:4d}  "
              f"close(<50cm): {n_close}")

    print("\n── Stopping ─────────────────────────────────────────────────────")
    mgr.stop()
    print(f"✓ Stopped.  Total scans received: {mgr.scan_count}")
    print("  LiDAR test complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_import()

    port = sys.argv[1] if len(sys.argv) > 1 else ""

    try:
        test_hardware(port)
    except KeyboardInterrupt:
        print("\nInterrupted.")
