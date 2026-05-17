#ifndef LIDAR_MANAGER_HPP
#define LIDAR_MANAGER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <map>
#include <condition_variable>
#include <chrono>
#include "CYdLidar.h"
#include "core/common/ydlidar_help.h"

namespace lidar {

  struct Point {
    float angle_deg;
    float distance_cm;
    float intensity;
  };

  struct Scan {
    std::vector<Point> points;
    uint64_t           timestamp_ns = 0;
    uint32_t           scan_index   = 0;
  };

  struct Config {
    std::string port = "";  // Empty for automatic detection
    int baudrate = 230400;  
    float frequency = 10.0f;
    float min_angle = -180.0f;
    float max_angle = 180.0f;
    float min_dist_cm = 5.0f;    
    float max_dist_cm = 6400.0f; 
  };

  class LidarManager {
  private:
    CYdLidar laser_;
    Config config_;
    std::thread worker_thread_;
    
    std::atomic<bool> is_running_{false};
    std::atomic<bool> has_error_{false};
    std::string last_error_msg_ = "";
    
    std::mutex data_mutex_;
    std::condition_variable cv_;
    Scan current_scan_;
    uint32_t scan_counter_{0};

    void run_loop() {
      LaserScan raw_scan;
      while (is_running_ && ydlidar::os_isOk()) {
	if (laser_.doProcessSimple(raw_scan)) {
	  Scan processed_frame;
	  processed_frame.timestamp_ns = raw_scan.stamp;
                
	  for (const auto& p : raw_scan.points) {
	    float angle = p.angle * 180.0f / M_PI;
	    float distance = p.range * 100.0f; 

	    if (distance <= config_.min_dist_cm || distance > config_.max_dist_cm) {
	      continue;
	    }
	    if (angle < config_.min_angle || angle > config_.max_angle) {
	      continue;
	    }

	    processed_frame.points.push_back({angle, distance, static_cast<float>(p.intensity)});
	  }

	  {
	    std::lock_guard<std::mutex> lock(data_mutex_);
	    processed_frame.scan_index = ++scan_counter_;
	    current_scan_ = std::move(processed_frame);
	  }
	  cv_.notify_all();
	} else {
	  std::lock_guard<std::mutex> lock(data_mutex_);
	  has_error_ = true;
	  last_error_msg_ = "Failed to capture telemetry frame.";
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }

  public:
    LidarManager() = default;
    
    ~LidarManager() {
      stop();
    }

    void configure(const Config& cfg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      config_ = cfg;
      ydlidar::os_init();

      std::string final_port = config_.port;
        
      if (final_port.empty() || final_port == "/dev/ttyUSB0") { 
	std::map<std::string, std::string> ports = ydlidar::lidarPortList();
	if (!ports.empty()) {
	  final_port = ports.begin()->second;
	  std::cout << "[LidarManager] Auto-detected hardware port: " << final_port << std::endl;
	} else {
	  has_error_ = true;
	  last_error_msg_ = "No active LiDAR interface detected on system buses.";
	  return;
	}
      }

      laser_.setlidaropt(LidarPropSerialPort, final_port.c_str(), final_port.size());
        
      std::string ignore_array = "";
      laser_.setlidaropt(LidarPropIgnoreArray, ignore_array.c_str(), ignore_array.size());
      laser_.setlidaropt(LidarPropSerialBaudrate, &config_.baudrate, sizeof(int));

      int opt_tri = TYPE_TRIANGLE;
      laser_.setlidaropt(LidarPropLidarType, &opt_tri, sizeof(int));

      int opt_dev = YDLIDAR_TYPE_SERIAL;
      laser_.setlidaropt(LidarPropDeviceType, &opt_dev, sizeof(int));

      int sample_rate = 4; 
      laser_.setlidaropt(LidarPropSampleRate, &sample_rate, sizeof(int));
      laser_.setlidaropt(LidarPropAbnormalCheckCount, &sample_rate, sizeof(int));

      int intensity_bit = 8;
      laser_.setlidaropt(LidarPropIntenstiyBit, &intensity_bit, sizeof(int));

      bool b_true = true;
      bool b_false = false;
      laser_.setlidaropt(LidarPropFixedResolution, &b_true, sizeof(bool));
      laser_.setlidaropt(LidarPropReversion, &b_false, sizeof(bool));
      laser_.setlidaropt(LidarPropInverted, &b_false, sizeof(bool));
      laser_.setlidaropt(LidarPropAutoReconnect, &b_true, sizeof(bool));
      laser_.setlidaropt(LidarPropSingleChannel, &b_false, sizeof(bool));
      laser_.setlidaropt(LidarPropIntenstiy, &b_true, sizeof(bool));
      laser_.setlidaropt(LidarPropSupportMotorDtrCtrl, &b_false, sizeof(bool));
      laser_.setlidaropt(LidarPropSupportHeartBeat, &b_false, sizeof(bool));

      laser_.setlidaropt(LidarPropMaxAngle, &config_.max_angle, sizeof(float));
      laser_.setlidaropt(LidarPropMinAngle, &config_.min_angle, sizeof(float));
        
      float max_range_m = config_.max_dist_cm / 100.0f;
      float min_range_m = config_.min_dist_cm / 100.0f;
      laser_.setlidaropt(LidarPropMaxRange, &max_range_m, sizeof(float));
      laser_.setlidaropt(LidarPropMinRange, &min_range_m, sizeof(float));
      laser_.setlidaropt(LidarPropScanFrequency, &config_.frequency, sizeof(float));

      laser_.enableGlassNoise(false);
      laser_.enableSunNoise(false);

      if (!laser_.initialize()) {
	has_error_ = true;
	last_error_msg_ = "Initialization rejected: " + std::string(laser_.DescribeError());
      }
    }

    bool start() {
      if (is_running_) return true;
      if (has_error_) return false;
        
      if (!laser_.turnOn()) {
	has_error_ = true;
	last_error_msg_ = "Power-on spin state failed: " + std::string(laser_.DescribeError());
	return false;
      }

      is_running_ = true;
      worker_thread_ = std::thread(&LidarManager::run_loop, this);
      return true;
    }

    void stop() {
      if (is_running_) {
	is_running_ = false;
	cv_.notify_all();
	if (worker_thread_.joinable()) {
	  worker_thread_.join();
	}
	laser_.turnOff();
	laser_.disconnecting();
      }
    }

    // Restored exact multi-argument function for binding.cpp
    Scan fetch_scan(bool wait_for_fresh = false, int timeout_ms = 150) {
      std::unique_lock<std::mutex> lock(data_mutex_);
      if (wait_for_fresh && is_running_ && !has_error_) {
	uint32_t current_idx = scan_counter_;
	cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, current_idx]() {
	  return scan_counter_ > current_idx || !is_running_ || has_error_;
	});
      }
      return current_scan_;
    }

    // Exposed methods for Pybind11 properties
    bool is_running() const { return is_running_.load(); }
    bool has_error() const { return has_error_.load(); }
    std::string last_error() const { return last_error_msg_; }
    uint32_t scan_count() const {
      return scan_counter_;
    }

    // Add this inside the public: block of class LidarManager in lidar_manager.hpp
    static uint64_t now_ns() {
      return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    }
  };

} // namespace lidar

#endif // LIDAR_MANAGER_HPP
