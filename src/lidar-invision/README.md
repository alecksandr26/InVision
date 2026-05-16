# lidar_invision — YDLIDAR Python bindings

Rolling-buffer LiDAR module for Raspberry Pi.  
Designed to pair with an AI vision pipeline: the LiDAR runs on its own background thread and you snapshot it in the same instant you grab a camera frame.

## How it works

```
LiDAR thread  ──► [rolling buffer] ──► fetch_scan()  ──► numpy array
                   (mutex-guarded)      (< 100 µs)        shape (N, 3)
                   always fresh                           [angle, dist_cm, intensity]
```

The background thread runs `doProcessSimple()` continuously at 10 Hz.  
Every complete scan atomically replaces the buffer and wakes any waiting callers.  
`fetch_scan()` just locks, copies, and unlocks — it never stalls your inference loop.

---

## Prerequisites

### 1. YDLIDAR SDK

```bash
git clone https://github.com/YDLIDAR/YDLidar-SDK.git
cd YDLidar-SDK
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON ..
make -j$(nproc)
sudo make install
```

> `CYdLidar.h` is **bundled in `src/`** — you do not need to copy any headers manually. The SDK install only provides the compiled library (`libydlidar_sdk.so`) that we link against.

### 2. Python dependencies

```bash
pip install pybind11 numpy
```

---

## Build & install

```bash
cd lidar_module
pip install .
```

Or for development (editable, no reinstall needed after source changes):

```bash
pip install -e .
```

---

## Quick start

```python
from lidar_sdk import LidarManager

mgr = LidarManager()
mgr.set_port("/dev/ttyUSB0")   # or leave empty for auto-detect
mgr.set_frequency(10.0)
mgr.set_distance_range(5.0, 150.0)   # cm

mgr.start()

# In your frame loop:
frame = camera.read()
scan  = mgr.fetch_scan()          # ~10–50 µs

pts = scan.points                 # numpy float32 (N, 3)
# pts[:, 0] → angle_deg
# pts[:, 1] → distance_cm
# pts[:, 2] → intensity

mgr.stop()
```

---

## API reference

### `LidarManager`

| Method / Property | Description |
|---|---|
| `set_port(port)` | Serial port string, e.g. `"/dev/ttyUSB0"` |
| `set_frequency(hz)` | Scan frequency in Hz (default `10.0`) |
| `set_angle_range(min_deg, max_deg)` | Angular FOV filter (default `-180, 180`) |
| `set_distance_range(min_cm, max_cm)` | Distance filter in cm (default `5, 150`) |
| `configure(config)` | Apply a `Config` object |
| `start()` | Start background thread. Idempotent. |
| `stop()` | Stop thread and disconnect. Idempotent. |
| `fetch_scan(wait_for_fresh, timeout_ms)` | Return latest scan snapshot |
| `is_running` | `True` if thread is active |
| `has_error` | `True` if a fatal error occurred |
| `last_error` | Error message string |
| `scan_count` | Total scans received since `start()` |
| `scan_age_ms` | Age of most recent scan in milliseconds |

### `fetch_scan(wait_for_fresh=False, timeout_ms=150)`

| `wait_for_fresh` | Behaviour |
|---|---|
| `False` (default) | Return immediately — grab what's in the buffer right now. Use this in your frame loop. |
| `True` | Block until a scan that started *after* this call is available. Use when you need scan ↔ frame temporal alignment. |

### `ScanResult`

```python
scan.points        # numpy float32, shape (N, 3): [angle_deg, dist_cm, intensity]
scan.timestamp_ns  # steady_clock nanoseconds when scan completed
scan.scan_index    # monotonic counter
scan.age_ms        # staleness in ms at moment fetch_scan() returned
```

---

## Typical AI pipeline pattern

```python
mgr.start()

while True:
    # ① Grab camera frame and LiDAR scan at the same moment
    frame = camera.read()
    scan  = mgr.fetch_scan()          # non-blocking, < 100 µs

    # ② Run inference (~100 ms)
    detections = model.infer(frame)

    # ③ Fuse — scan is already in memory, no waiting
    pts = scan.points
    close = pts[pts[:, 1] < 50]       # points within 50 cm
    # ... spatial fusion logic ...

mgr.stop()
```

---

## Validate install

```bash
python lidar_test.py              # no hardware — module smoke test only
python lidar_test.py /dev/ttyUSB0 # full hardware test
```
