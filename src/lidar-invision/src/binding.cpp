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
PYBIND11_MODULE(lidar_invision, m) {
    m.doc() = "YDLIDAR rolling-buffer Python bindings (pybind11)";

    // 1. Bind Config Structure
    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("port",        &Config::port)
        .def_readwrite("baudrate",    &Config::baudrate)
        .def_readwrite("frequency",   &Config::frequency)
        .def_readwrite("min_angle",   &Config::min_angle)
        .def_readwrite("max_angle",   &Config::max_angle)
        .def_readwrite("min_dist_cm", &Config::min_dist_cm)
        .def_readwrite("max_dist_cm", &Config::max_dist_cm);

    // 2. Bind the correct C++ ScanResult data context (Fixes age_ms mismatch)
    py::class_<ScanResult>(m, "ScanResult")
        .def_readwrite("points",       &ScanResult::points)
        .def_readwrite("timestamp_ns", &ScanResult::timestamp_ns)
        .def_readwrite("scan_index",   &ScanResult::scan_index)
        .def_readwrite("age_ms",       &ScanResult::age_ms);

    // 3. Bind Main LidarManager Class
    py::class_<LidarManager>(m, "LidarManager")
        .def(py::init<>())
        // Explicit lambda captures configuration values cleanly without slicing
        .def("configure", [](LidarManager& self, const Config& cfg) {
            self.configure(cfg);
        }, py::arg("cfg"), "Apply setup configuration struct")
        .def("start",     &LidarManager::start,     "Start background telemetry scanning thread")
        .def("stop",      &LidarManager::stop,      "Stop telemetry processing thread")
        .def_property_readonly("is_running", &LidarManager::is_running)
        .def_property_readonly("has_error",  &LidarManager::has_error)
        .def_property_readonly("last_error", &LidarManager::last_error)
        .def_property_readonly("scan_count", &LidarManager::scan_count)
        .def("fetch_scan", [](LidarManager& self, bool wait_for_fresh, int timeout_ms) {
            Scan scan;
            {
                py::gil_scoped_release release;
                scan = self.fetch_scan(wait_for_fresh, timeout_ms);
            }
            // Transform the raw Scan into the bounded ScanResult structure
            return make_result(std::move(scan));
        }, py::arg("wait_for_fresh") = false, py::arg("timeout_ms") = 150);
}
