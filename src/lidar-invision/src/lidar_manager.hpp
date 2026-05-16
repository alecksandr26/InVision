#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "CYdLidar.h"   // bundled in src/ — SDK internals still needed at link time

namespace lidar {

  // ─────────────────────────────────────────────
  //  A single LiDAR point returned to Python
  // ─────────────────────────────────────────────
  struct Point {
    float angle_deg;     // degrees,  -180 … +180
    float distance_cm;   // centimetres
    float intensity;     // 0 … 255  (0 if sensor has no intensity)
  };

  // ─────────────────────────────────────────────
  //  One complete scan snapshot
  // ─────────────────────────────────────────────
  struct Scan {
    std::vector<Point> points;
    uint64_t           timestamp_ns = 0;   // system clock, nanoseconds
    uint32_t           scan_index   = 0;   // monotonically increasing
  };

  // ─────────────────────────────────────────────
  //  Runtime configuration (all fields optional;
  //  defaults match YDLIDAR T-mini)
  // ─────────────────────────────────────────────
  struct Config {
    std::string port        = "";       // empty = auto-detect
    int         baudrate    = 230400;
    float       frequency   = 10.0f;   // Hz
    float       min_angle   = -180.0f; // degrees
    float       max_angle   =  180.0f;
    float       min_dist_cm =    5.0f;
    float       max_dist_cm = 1500.0f;
    bool        intensity   = true;
    bool        auto_reconnect = true;
  };

  // ─────────────────────────────────────────────
  //  LidarManager
  //
  //  Usage (Python side mirrors this exactly):
  //
  //    mgr = LidarManager()
  //    mgr.configure(port="/dev/ttyUSB0", frequency=10.0)
  //    mgr.start()
  //
  //    # In your frame loop:
  //    frame   = camera.read()
  //    scan    = mgr.fetch_scan()          # ~0 µs cost
  //    # or:
  //    scan    = mgr.fetch_scan(wait_for_fresh=True, timeout_ms=120)
  //
  //    mgr.stop()
  // ─────────────────────────────────────────────
  class LidarManager {
  public:
    LidarManager()  = default;
    ~LidarManager() { stop(); }

    // Disallow copy/move — owns hardware resource
    LidarManager(const LidarManager&)            = delete;
    LidarManager& operator=(const LidarManager&) = delete;

    // ── Configuration ──────────────────────────────────────────────────
    // Call before start().  Safe to call again after stop() to reconfigure.
    void configure(const Config& cfg) {
      if (running_.load()) {
	throw std::runtime_error("Cannot reconfigure while running. Call stop() first.");
      }
      config_ = cfg;
    }

    // Individual setters — convenient from Python keyword args
    void set_port(const std::string& p)  { config_.port = p; }
    void set_baudrate(int b)             { config_.baudrate = b; }
    void set_frequency(float f)          { config_.frequency = f; }
    void set_angle_range(float mn, float mx) {
      config_.min_angle = mn;
      config_.max_angle = mx;
    }
    void set_distance_range(float mn_cm, float mx_cm) {
      config_.min_dist_cm = mn_cm;
      config_.max_dist_cm = mx_cm;
    }

    // ── Lifecycle ──────────────────────────────────────────────────────
    void start() {
      if (running_.load()) return;   // idempotent

      running_.store(true);
      error_flag_.store(false);
      last_error_.clear();

      worker_ = std::thread(&LidarManager::worker_loop, this);
    }

    void stop() {
      if (!running_.load()) return;  // idempotent

      running_.store(false);
      cv_.notify_all();              // wake any blocked fetch_scan()

      if (worker_.joinable()) worker_.join();
    }

    bool is_running() const { return running_.load(); }
    bool has_error()  const { return error_flag_.load(); }

    std::string last_error() const {
      std::lock_guard<std::mutex> lk(meta_mutex_);
      return last_error_;
    }

    // ── Data access ────────────────────────────────────────────────────

    // Returns a *copy* of the most recent complete scan.
    // Cost: one mutex lock + memcpy of ~N×3 floats.  Typically < 50 µs.
    //
    // wait_for_fresh:  if true, blocks until a scan that arrived *after*
    //                  this call began is available (or timeout expires).
    //                  Use this when you want the scan aligned to the
    //                  current camera frame rather than the previous one.
    //
    // timeout_ms:  maximum wait in milliseconds (only used when
    //              wait_for_fresh = true).  Returns last known scan on
    //              timeout rather than throwing.
    Scan fetch_scan(bool wait_for_fresh = false, int timeout_ms = 150) {
      if (wait_for_fresh) {
	// Capture the scan index *before* we start waiting so we know
	// when a genuinely newer scan has arrived.
	uint32_t seen_index;
	{
	  std::lock_guard<std::mutex> lk(scan_mutex_);
	  seen_index = current_scan_.scan_index;
	}

	std::unique_lock<std::mutex> lk(scan_mutex_);
	cv_.wait_for(lk,
		     std::chrono::milliseconds(timeout_ms),
		     [this, seen_index]() {
		       return !running_.load() ||
			 current_scan_.scan_index > seen_index;
		     });
	return current_scan_;   // copy under lock
      }

      // Default fast path — just copy whatever is there right now
      std::lock_guard<std::mutex> lk(scan_mutex_);
      return current_scan_;
    }

    // Convenience: age of the most recent scan in milliseconds
    double scan_age_ms() const {
      uint64_t ts;
      {
	std::lock_guard<std::mutex> lk(scan_mutex_);
	ts = current_scan_.timestamp_ns;
      }
      if (ts == 0) return -1.0;
      auto now = std::chrono::steady_clock::now();
      uint64_t now_ns = static_cast<uint64_t>(
					      std::chrono::duration_cast<std::chrono::nanoseconds>(
												   now.time_since_epoch()).count());
      return static_cast<double>(now_ns - ts) / 1e6;
    }

    // Total scans received since start()
    uint32_t scan_count() const {
      std::lock_guard<std::mutex> lk(scan_mutex_);
      return current_scan_.scan_index;
    }

    const Config& config() const { return config_; }

  private:
    // ── Background worker ──────────────────────────────────────────────
    void worker_loop() {
      CYdLidar laser;

      // ── Apply configuration ──────────────────────────────────────
      std::string port = config_.port;
      if (port.empty()) {
	auto ports = ydlidar::lidarPortList();
	if (!ports.empty()) {
	  port = ports.begin()->second;
	} else {
	  set_error("No LiDAR port detected and none configured.");
	  return;
	}
      }

      laser.setlidaropt(LidarPropSerialPort,
			port.c_str(), static_cast<int>(port.size()));

      std::string ignore_array;
      laser.setlidaropt(LidarPropIgnoreArray,
			ignore_array.c_str(), static_cast<int>(ignore_array.size()));

      int iv = config_.baudrate;
      laser.setlidaropt(LidarPropSerialBaudrate, &iv, sizeof(int));

      iv = TYPE_TRIANGLE;
      laser.setlidaropt(LidarPropLidarType, &iv, sizeof(int));

      iv = YDLIDAR_TYPE_SERIAL;
      laser.setlidaropt(LidarPropDeviceType, &iv, sizeof(int));

      iv = 4;   // sample rate for T-mini
      laser.setlidaropt(LidarPropSampleRate, &iv, sizeof(int));

      iv = 4;
      laser.setlidaropt(LidarPropAbnormalCheckCount, &iv, sizeof(int));

      iv = 8;
      laser.setlidaropt(LidarPropIntenstiyBit, &iv, sizeof(int));

      bool bv = true;
      laser.setlidaropt(LidarPropFixedResolution,  &bv, sizeof(bool));
      bv = false;
      laser.setlidaropt(LidarPropReversion,        &bv, sizeof(bool));
      laser.setlidaropt(LidarPropInverted,         &bv, sizeof(bool));

      bv = config_.auto_reconnect;
      laser.setlidaropt(LidarPropAutoReconnect,    &bv, sizeof(bool));

      bv = false;
      laser.setlidaropt(LidarPropSingleChannel,    &bv, sizeof(bool));

      bv = config_.intensity;
      laser.setlidaropt(LidarPropIntenstiy,        &bv, sizeof(bool));

      bv = false;
      laser.setlidaropt(LidarPropSupportMotorDtrCtrl, &bv, sizeof(bool));
      laser.setlidaropt(LidarPropSupportHeartBeat,    &bv, sizeof(bool));

      float fv = config_.max_angle;
      laser.setlidaropt(LidarPropMaxAngle, &fv, sizeof(float));
      fv = config_.min_angle;
      laser.setlidaropt(LidarPropMinAngle, &fv, sizeof(float));

      fv = config_.max_dist_cm / 100.0f;   // SDK wants metres
      laser.setlidaropt(LidarPropMaxRange, &fv, sizeof(float));
      fv = config_.min_dist_cm / 100.0f;
      laser.setlidaropt(LidarPropMinRange, &fv, sizeof(float));

      fv = config_.frequency;
      laser.setlidaropt(LidarPropScanFrequency, &fv, sizeof(float));

      laser.enableGlassNoise(false);
      laser.enableSunNoise(false);

      // ── Connect ──────────────────────────────────────────────────
      if (!laser.initialize()) {
	set_error(std::string("LiDAR initialize failed: ") +
		  laser.DescribeError());
	return;
      }
      if (!laser.turnOn()) {
	set_error(std::string("LiDAR turnOn failed: ") +
		  laser.DescribeError());
	return;
      }

      // ── Scan loop ─────────────────────────────────────────────────
      LaserScan raw;
      uint32_t  index = 0;

      while (running_.load()) {
	if (!laser.doProcessSimple(raw)) {
	  // Transient failure — log but keep running
	  set_error(std::string("doProcessSimple failed: ") +
		    laser.DescribeError(), /*fatal=*/false);
	  continue;
	}

	// Build our clean scan
	Scan next;
	next.scan_index   = ++index;
	next.timestamp_ns = now_ns();
	next.points.reserve(raw.points.size());

	for (const auto& p : raw.points) {
	  float dist_cm = p.range * 100.0f;
	  float angle   = p.angle * 180.0f / static_cast<float>(M_PI);

	  // Distance filter (noise)
	  if (dist_cm < config_.min_dist_cm ||
	      dist_cm > config_.max_dist_cm) continue;

	  // Angular FOV filter
	  if (angle < config_.min_angle ||
	      angle > config_.max_angle)  continue;

	  next.points.push_back({angle, dist_cm,
	      static_cast<float>(p.intensity)});
	}

	// Swap into shared buffer and wake any waiters
	{
	  std::lock_guard<std::mutex> lk(scan_mutex_);
	  current_scan_ = std::move(next);
	}
	cv_.notify_all();
      }

      // ── Shutdown ──────────────────────────────────────────────────
      laser.turnOff();
      laser.disconnecting();
    }

    // ── Helpers ────────────────────────────────────────────────────────
    static uint64_t now_ns() {
      return static_cast<uint64_t>(
				   std::chrono::duration_cast<std::chrono::nanoseconds>(
											std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    void set_error(const std::string& msg, bool fatal = true) {
      {
	std::lock_guard<std::mutex> lk(meta_mutex_);
	last_error_ = msg;
      }
      if (fatal) {
	error_flag_.store(true);
	running_.store(false);
	cv_.notify_all();
      }
    }

    // ── State ──────────────────────────────────────────────────────────
    Config config_;

    std::thread             worker_;
    std::atomic<bool>       running_    {false};
    std::atomic<bool>       error_flag_ {false};

    mutable std::mutex      scan_mutex_;
    std::condition_variable cv_;
    Scan                    current_scan_;          // guarded by scan_mutex_

    mutable std::mutex      meta_mutex_;
    std::string             last_error_;            // guarded by meta_mutex_
  };
} // namespace lidar
