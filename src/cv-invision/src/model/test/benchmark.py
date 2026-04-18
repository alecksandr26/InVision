import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time
import psutil
import argparse
import threading
import numpy as np
import cProfile
import pstats
from io import StringIO
from pathlib import Path

from src.model.detection_model import load_detection_model_test, load_detection_model_prod
from src.utils.log              import get_logger

logger = get_logger(__name__)

# ============================================================
# SYSTEM MONITOR — runs in background thread
# ============================================================

class SystemMonitor:
    """
    Monitors CPU and RAM usage in a background thread.
    Call start() before inference and stop() after to collect metrics.
    """

    def __init__(self, interval: float = 0.5):
        self.interval    = interval
        self.cpu_samples = []
        self.ram_samples = []
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._monitor, daemon=True)
        self.process     = psutil.Process(os.getpid())

    def _monitor(self):
        while not self._stop_event.is_set():
            self.cpu_samples.append(self.process.cpu_percent(interval=None))
            self.ram_samples.append(self.process.memory_info().rss / 1024**2)  # MB
            time.sleep(self.interval)

    def start(self):
        logger.info("📊 System monitor started...")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        logger.info("📊 System monitor stopped.")

    def report(self):
        if not self.cpu_samples or not self.ram_samples:
            logger.warning("No samples collected!")
            return

        logger.info("=" * 55)
        logger.info("📊 SYSTEM USAGE REPORT")
        logger.info("=" * 55)
        logger.info(f"  CPU avg : {sum(self.cpu_samples) / len(self.cpu_samples):.1f}%")
        logger.info(f"  CPU max : {max(self.cpu_samples):.1f}%")
        logger.info(f"  RAM avg : {sum(self.ram_samples) / len(self.ram_samples):.1f} MB")
        logger.info(f"  RAM max : {max(self.ram_samples):.1f} MB")
        logger.info(f"  Samples : {len(self.cpu_samples)}")
        logger.info("=" * 55)


# ============================================================
# WARMUP
# ============================================================

def warmup(model, runs: int = 3):
    """
    Warms up the model with dummy frames before benchmarking.
    This ensures the first real inference is not penalized by
    lazy initialization or JIT compilation overhead.

    Args:
        model: YOLO model instance
        runs : Number of warmup runs (default: 3)
    """
    logger.info(f"🔥 Warming up model ({runs} runs)...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

    for i in range(runs):
        model(dummy)
        logger.debug(f"  Warmup run {i + 1}/{runs} done")

    logger.info("✅ Warmup complete!")


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(model, image: np.ndarray, runs: int = 20) -> dict:
    """
    Runs inference N times and collects timing metrics.

    Args:
        model : YOLO model instance
        image : Input image as numpy array
        runs  : Number of benchmark runs (default: 20)

    Returns:
        dict with min, max, avg, fps metrics
    """
    logger.info(f"⏱️  Running benchmark ({runs} runs)...")
    times = []

    for i in range(runs):
        start = time.perf_counter()
        model(image)
        end   = time.perf_counter()

        elapsed = (end - start) * 1000  # ms
        times.append(elapsed)
        logger.debug(f"  Run {i + 1:02d}/{runs} — {elapsed:.1f} ms")

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    fps    = 1000 / avg_ms

    return {
        "runs"   : runs,
        "avg_ms" : avg_ms,
        "min_ms" : min_ms,
        "max_ms" : max_ms,
        "fps"    : fps,
        "times"  : times,
    }


def print_benchmark_report(label: str, metrics: dict):
    logger.info("=" * 55)
    logger.info(f"⏱️  BENCHMARK REPORT — {label}")
    logger.info("=" * 55)
    logger.info(f"  Runs    : {metrics['runs']}")
    logger.info(f"  Avg     : {metrics['avg_ms']:.1f} ms")
    logger.info(f"  Min     : {metrics['min_ms']:.1f} ms")
    logger.info(f"  Max     : {metrics['max_ms']:.1f} ms")
    logger.info(f"  FPS     : {metrics['fps']:.1f}")
    logger.info("=" * 55)


# ============================================================
# COMPARE TEST vs PROD
# ============================================================

def run_comparison(image: np.ndarray, runs: int = 20, warmup_runs: int = 3):
    """
    Loads both TEST and PROD models, warms them up, benchmarks them,
    and prints a side by side comparison report.

    Args:
        image       : Input image as numpy array
        runs        : Number of benchmark runs per model
        warmup_runs : Number of warmup runs per model
    """
    results = {}

    for label, loader in [("TEST", load_detection_model_test), ("PROD", load_detection_model_prod)]:
        logger.info(f"\n{'=' * 55}")
        logger.info(f"🤖 Loading {label} model...")
        logger.info(f"{'=' * 55}")

        model   = loader()
        monitor = SystemMonitor()

        warmup(model, runs=warmup_runs)

        monitor.start()
        metrics = benchmark(model, image, runs=runs)
        monitor.stop()

        print_benchmark_report(label, metrics)
        monitor.report()

        results[label] = {"metrics": metrics}

    # Side by side comparison
    logger.info("\n" + "=" * 55)
    logger.info("📊 TEST vs PROD COMPARISON")
    logger.info("=" * 55)
    for key in ["avg_ms", "min_ms", "max_ms", "fps"]:
        test_val = results["TEST"]["metrics"][key]
        prod_val = results["PROD"]["metrics"][key]
        unit     = "FPS" if key == "fps" else "ms"
        logger.info(f"  {key:<8} TEST={test_val:.1f} {unit}  |  PROD={prod_val:.1f} {unit}")
    logger.info("=" * 55)


# ============================================================
# SINGLE MODEL BENCHMARK
# ============================================================

def run_single(model_type: str, image: np.ndarray, runs: int = 20, warmup_runs: int = 3):
    """
    Benchmarks a single model (test or prod).

    Args:
        model_type  : "test" or "prod"
        image       : Input image as numpy array
        runs        : Number of benchmark runs
        warmup_runs : Number of warmup runs
    """
    loader = load_detection_model_test if model_type == "test" else load_detection_model_prod
    label  = model_type.upper()

    logger.info(f"🤖 Loading {label} model...")
    model   = loader()
    monitor = SystemMonitor()

    warmup(model, runs=warmup_runs)

    monitor.start()
    metrics = benchmark(model, image, runs=runs)
    monitor.stop()

    print_benchmark_report(label, metrics)
    monitor.report()


# ============================================================
# ARGS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="📊 InVision Detection Model — Performance Benchmark"
    )
    parser.add_argument(
        "--input", "-i",
        type     = str,
        required = True,
        help     = "Path to input image for benchmarking"
    )
    parser.add_argument(
        "--mode", "-m",
        type    = str,
        default = "compare",
        choices = ["test", "prod", "compare"],
        help    = "Benchmark mode: test | prod | compare (default: compare)"
    )
    parser.add_argument(
        "--runs", "-r",
        type    = int,
        default = 20,
        help    = "Number of benchmark runs (default: 20)"
    )
    parser.add_argument(
        "--warmup", "-w",
        type    = int,
        default = 3,
        help    = "Number of warmup runs (default: 3)"
    )
    parser.add_argument(
        "--profile",
        action  = "store_true",
        help    = "Run benchmark under cProfile and save stats to profile.out"
    )
    return parser.parse_args()


# ============================================================
# MAIN (with profiling support)
# ============================================================

def _run_benchmark(args):
    """
    Core benchmarking logic (separated to be profiled).
    """
    image = cv2.imread(args.input)
    if image is None:
        logger.error(f"Could not read image: '{args.input}'")
        raise SystemExit(1)

    logger.info(f"🚀 Starting InVision benchmark pipeline...")
    logger.info(f"⚙️  Mode    : {args.mode}")
    logger.info(f"⚙️  Runs    : {args.runs}")
    logger.info(f"⚙️  Warmup  : {args.warmup}")
    logger.info(f"⚙️  Image   : {args.input}")

    if args.mode == "compare":
        run_comparison(image, runs=args.runs, warmup_runs=args.warmup)
    else:
        run_single(args.mode, image, runs=args.runs, warmup_runs=args.warmup)


def main():
    args = parse_args()
    if args.profile:
        logger.info("📈 Running under cProfile...")
        profiler = cProfile.Profile()
        profiler.enable()
        _run_benchmark(args)
        profiler.disable()
        # Save raw stats
        profiler.dump_stats("profile.out")
        # Print top 30 functions by cumulative time
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
        stats.print_stats(30)
        logger.info("\n📈 cProfile output (top 30 cumulative):\n" + stream.getvalue())
    else:
        _run_benchmark(args)


if __name__ == "__main__":
    main()
