#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <iostream>

#include "CYdLidar.h"

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
    std::string port        = "/dev/ttyUSB0";       
    int         baudrate    = 230400;
    float       frequency   = 10.0f;   
    float       min_angle   = -180.0f; 
    float       max_angle   =  180.0f; 
    float       min_dist_cm = 5.0f;    
    float       max_dist_cm = 150.0f;  
  };

  class LidarManager {
  public:
    LidarManager() = default;

    // Fixes the "terminate called without an active exception" crash
    ~LidarManager() {
      stop();
      if (worker_.joinable()) {
        worker_.join();
      }
    }

    void configure(const Config& cfg) {
      std::lock_guard<std::mutex> lk(meta_mutex_);
      config_ = cfg;
    }

    Config get_config() const {
      std::lock_guard<std::mutex> lk(meta_mutex_);
      return config_;
    }

    void start() {
      if (running_.load()) return;
      running_.store(true);
      error_flag_.store(false);
      worker_ = std::thread(&LidarManager::run_loop, this);
    }

    void stop() {
      running_.store(false);
      cv_.notify_all();
    }

    bool is_running() const { return running_.load(); }
    bool has_error() const { return error_flag_.load(); }
    
    std::string last_error() const {
      std::lock_guard<std::mutex> lk(meta_mutex_);
      return last_error_;
    }

    uint32_t scan_count() const {
      std::lock_guard<std::mutex> lk(scan_mutex_);
      return current_scan_.scan_index;
    }

    Scan fetch_scan(bool wait_for_fresh = false, int timeout_ms = 150) {
      std::unique_lock<std::mutex> lk(scan_mutex_);
      if (wait_for_fresh) {
        uint32_t old_idx = current_scan_.scan_index;
        cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this, old_idx] {
          return current_scan_.scan_index > old_idx || !running_.load();
        });
      }
      return current_scan_;
    }

  private:
    // ── The missing method for your T-mini ──
    bool initialize_sdk(CYdLidar& laser) {
      laser.setSerialPort(config_.port);
      laser.setSerialBaudrate(config_.baudrate);

      // T-mini uses TOF type (constant 3) and standard serial communication
      laser.setLidarType(3);  
      laser.setDeviceType(YDLIDAR_TYPE_SERIAL);
      laser.setSampleRate(4); // 4kHz

      laser.setMaxAngle(config_.max_angle);
      laser.setMinAngle(config_.min_angle);
      laser.setMaxRange(config_.max_dist_cm / 100.0f); 
      laser.setMinRange(config_.min_dist_cm / 100.0f);
      laser.setScanFrequency(config_.frequency);

      laser.setAutoReconnect(true);
      laser.setSingleChannel(false); 

      return laser.initialize();
    }

    void run_loop() {
      CYdLidar laser;
      
      if (!initialize_sdk(laser)) {
        set_error("Failed to initialize YDLIDAR hardware settings.", true);
        return;
      }

      if (!laser.turnOn()) {
        set_error("Failed to spin up LiDAR motor.", true);
        return;
      }

      while (running_.load()) {
        LaserScan raw_scan;
        if (laser.doProcessSimple(raw_scan)) {
          Scan next;
          next.timestamp_ns = now_ns();
          
          {
            std::lock_guard<std::mutex> lk(scan_mutex_);
            next.scan_index = current_scan_.scan_index + 1;
          }

          for (const auto& p : raw_scan.points) {
            float angle = p.angle * 180.0f / M_PI; // convert radians to degrees
            if (angle > 180.0f) angle -= 360.0f;

            float dist_cm = p.range * 100.0f;
            if (dist_cm < config_.min_dist_cm || dist_cm > config_.max_dist_cm) continue;
            if (angle < config_.min_angle || angle > config_.max_angle) continue;

            next.points.push_back({angle, dist_cm, static_cast<float>(p.intensity)});
          }

          {
            std::lock_guard<std::mutex> lk(scan_mutex_);
            current_scan_ = std::move(next);
          }
          cv_.notify_all();
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }

      laser.turnOff();
      laser.disconnecting();
    }

    static uint64_t now_ns() {
      return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
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

    Config             config_;
    std::thread        worker_;
    std::atomic<bool>  running_    {false};
    std::atomic<bool>  error_flag_ {false};

    mutable std::mutex cv_m_;
    std::condition_variable cv_;

    mutable std::mutex meta_mutex_;
    std::string        last_error_ = "";

    mutable std::mutex scan_mutex_;
    Scan               current_scan_;
  };
}
