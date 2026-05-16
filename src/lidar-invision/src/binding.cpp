#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "lidar_manager.hpp"

namespace py = pybind11;
using namespace lidar;

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: convert Scan → numpy array  shape (N, 3)
//          columns: [angle_deg, distance_cm, intensity]
//
//  Returns a zero-copy numpy array backed by a freshly-allocated buffer.
//  The Scan is passed by value (already a copy from the shared buffer) so
//  we can safely move the points vector into the capsule.
// ─────────────────────────────────────────────────────────────────────────────
static py::array_t<float> scan_to_numpy(Scan scan) {
  const ssize_t n = static_cast<ssize_t>(scan.points.size());

  // Allocate a contiguous float buffer: n × 3
  auto result = py::array_t<float>({n, ssize_t(3)});
  auto buf    = result.request();
  float* ptr  = static_cast<float*>(buf.ptr);

  for (ssize_t i = 0; i < n; ++i) {
    ptr[i * 3 + 0] = scan.points[i].angle_deg;
    ptr[i * 3 + 1] = scan.points[i].distance_cm;
    ptr[i * 3 + 2] = scan.points[i].intensity;
  }
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Python-facing ScanResult — wraps Scan so Python gets both the numpy
//  array AND the metadata (timestamp, index, age) in one object.
// ─────────────────────────────────────────────────────────────────────────────
struct ScanResult {
  py::array_t<float> points;       // shape (N, 3): angle, dist_cm, intensity
  uint64_t           timestamp_ns;
  uint32_t           scan_index;
  double             age_ms;       // milliseconds since this scan was captured
};

static ScanResult make_result(Scan scan) {
  uint64_t ts  = scan.timestamp_ns;
  uint32_t idx = scan.scan_index;

  // Age calculation (steady_clock based, same as lidar_manager.hpp)
  double age = 0.0;
  if (ts > 0) {
    uint64_t now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    age = static_cast<double>(now - ts) / 1e6;
  }

  return ScanResult{
    scan_to_numpy(std::move(scan)),
    ts, idx, age
  };
}

// ─────────────────────────────────────────────────────────────────────────────
//  Module definition
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(lidar_sdk, m) {
  m.doc() = "YDLIDAR rolling-buffer Python bindings";

  // ── ScanResult ────────────────────────────────────────────────────────
  py::class_<ScanResult>(m, "ScanResult")
    .def_readonly("points",       &ScanResult::points,
		  "numpy float32 array, shape (N, 3): [angle_deg, dist_cm, intensity]")
    .def_readonly("timestamp_ns", &ScanResult::timestamp_ns,
		  "Monotonic timestamp (nanoseconds) when the scan completed")
    .def_readonly("scan_index",   &ScanResult::scan_index,
		  "Monotonically increasing scan counter since start()")
    .def_readonly("age_ms",       &ScanResult::age_ms,
		  "Age of this scan in milliseconds at the moment fetch_scan() returned")
    .def("__repr__", [](const ScanResult& r) {
      return "<ScanResult points=" + std::to_string(r.points.shape(0)) +
	" age_ms=" + std::to_string(r.age_ms) +
	" index=" + std::to_string(r.scan_index) + ">";
    });

  // ── Config ────────────────────────────────────────────────────────────
  py::class_<Config>(m, "Config")
    .def(py::init<>())
    .def_readwrite("port",           &Config::port)
    .def_readwrite("baudrate",       &Config::baudrate)
    .def_readwrite("frequency",      &Config::frequency)
    .def_readwrite("min_angle",      &Config::min_angle)
    .def_readwrite("max_angle",      &Config::max_angle)
    .def_readwrite("min_dist_cm",    &Config::min_dist_cm)
    .def_readwrite("max_dist_cm",    &Config::max_dist_cm)
    .def_readwrite("intensity",      &Config::intensity)
    .def_readwrite("auto_reconnect", &Config::auto_reconnect)
    .def("__repr__", [](const Config& c) {
      return "<Config port='" + c.port +
	"' freq=" + std::to_string(c.frequency) + "Hz>";
    });

  // ── LidarManager ─────────────────────────────────────────────────────
  py::class_<LidarManager>(m, "LidarManager")
    .def(py::init<>())

    // ── Configuration ──────────────────────────────────────────────
    .def("configure", &LidarManager::configure, py::arg("config"),
	 "Apply a Config object. Must be called before start().")

    .def("set_port", &LidarManager::set_port, py::arg("port"),
	 "Set serial port, e.g. '/dev/ttyUSB0'")
    .def("set_baudrate", &LidarManager::set_baudrate, py::arg("baudrate"))
    .def("set_frequency", &LidarManager::set_frequency, py::arg("hz"),
	 "Scan frequency in Hz (default 10.0)")
    .def("set_angle_range", &LidarManager::set_angle_range,
	 py::arg("min_deg"), py::arg("max_deg"),
	 "Filter points outside this angular window (degrees)")
    .def("set_distance_range", &LidarManager::set_distance_range,
	 py::arg("min_cm"), py::arg("max_cm"),
	 "Filter points outside this distance range (centimetres)")

    // ── Lifecycle ──────────────────────────────────────────────────
    .def("start", [](LidarManager& self) {
      // Release the GIL: start() spawns a thread and returns quickly,
      // but the OS thread creation can briefly block.
      py::gil_scoped_release release;
      self.start();
    }, "Start the background LiDAR thread. Idempotent.")

    .def("stop", [](LidarManager& self) {
      py::gil_scoped_release release;
      self.stop();
    }, "Stop the background thread and disconnect the LiDAR. Idempotent.")

    .def_property_readonly("is_running",  &LidarManager::is_running)
    .def_property_readonly("has_error",   &LidarManager::has_error)
    .def_property_readonly("last_error",  &LidarManager::last_error)
    .def_property_readonly("scan_count",  &LidarManager::scan_count,
			   "Total complete scans received since start()")
    .def_property_readonly("scan_age_ms", &LidarManager::scan_age_ms,
			   "Age of the most recent scan in milliseconds (-1 if no scan yet)")
    .def_property_readonly("config",      &LidarManager::config)

    // ── Core fetch — THE key method ────────────────────────────────
    .def("fetch_scan",
	 [](LidarManager& self, bool wait_for_fresh, int timeout_ms) {
	   Scan scan;
	   {
	     // Release GIL only on the blocking path.
	     // The fast path (wait_for_fresh=False) still briefly
	     // releases because the mutex acquisition is lock-based
	     // and we never want to hold the GIL while holding a
	     // C++ mutex.
	     py::gil_scoped_release release;
	     scan = self.fetch_scan(wait_for_fresh, timeout_ms);
	   }
	   return make_result(std::move(scan));
	 },
	 py::arg("wait_for_fresh") = false,
	 py::arg("timeout_ms")     = 150,
	 R"doc(
Fetch the most recent complete LiDAR scan.

Parameters
----------
wait_for_fresh : bool
    If False (default): return immediately with whatever is in the buffer.
    This is the recommended mode for AI pipelines — call it right after
    grabbing a camera frame and it costs < 100 µs.

    If True: block until a scan that started *after* this call began is
    available.  Use when you need the scan to be temporally aligned with
    the current frame rather than the previous one.

timeout_ms : int
    Maximum wait time in milliseconds when wait_for_fresh=True.
    If the timeout expires the last known scan is returned anyway.

Returns
-------
ScanResult
    .points      — numpy float32 (N, 3): [angle_deg, dist_cm, intensity]
    .timestamp_ns — when the scan was captured (steady_clock nanoseconds)
    .scan_index  — monotonic counter
    .age_ms      — staleness in milliseconds at the moment of return
)doc")

    .def("__repr__", [](const LidarManager& self) {
      return std::string("<LidarManager running=") +
	(self.is_running() ? "True" : "False") + ">";
    });
}
