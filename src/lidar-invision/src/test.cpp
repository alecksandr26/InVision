#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include "lidar_manager.hpp"

using namespace lidar;
using namespace std;

int main() {
  cout << "=== Initializing Updated LidarManager ===" << endl;

  LidarManager manager;
  Config cfg;

  // Leaving port empty triggers automatic detection inside our updated class
  cfg.port = "";  
  cfg.baudrate = 230400;
  cfg.frequency = 10.0f;

  // 1. Call configure (void function)
  manager.configure(cfg);

  // 2. Check if configuration encountered an auto-detect error
  if (manager.has_error()) {
    cerr << "Configuration Error: " << manager.last_error() << endl;
    return -1;
  }

  cout << "Starting thread worker loop..." << endl;
  if (!manager.start()) {
    cerr << "Startup Error: " << manager.last_error() << endl;
    return -1;
  }

  // Allow hardware spin-up safety clearance time window
  std::this_thread::sleep_for(std::chrono::seconds(2));

  cout << "\nPolling real-time environment data frames:" << endl;
  cout << string(50, '-') << endl;

  for (int i = 0; i < 10; ++i) {
    if (manager.has_error()) {
      cerr << "Runtime failure noticed: " << manager.last_error() << endl;
      break;
    }

    // Passes false to non-blocking fetch frame quickly for testing
    Scan frame = manager.fetch_scan(false);
        
    if (frame.points.empty()) {
      cout << "Waiting for valid sweep points to clear..." << endl;
    } else {
      cout << "Frame #" << frame.scan_index 
	   << " | Total Points mapped: " << frame.points.size() 
	   << " | First reading: " << fixed << setprecision(2)
	   << frame.points[0].angle_deg << " deg, " 
	   << frame.points[0].distance_cm << " cm" << endl;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  cout << "\nTearing down thread cycles..." << endl;
  manager.stop();
  cout << "Done." << endl;

  return 0;
}
