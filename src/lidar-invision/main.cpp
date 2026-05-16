#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <unistd.h>  // 🔥 para usleep
#include "CYdLidar.h"
#include "core/common/ydlidar_help.h"

using namespace std;
using namespace ydlidar;
using namespace ydlidar::core::common;

int main(int argc, char *argv[])
{
  printLogo();
  os_init();

  std::string port;
  std::map<std::string, std::string> ports = ydlidar::lidarPortList();
  std::map<std::string, std::string>::iterator it;

  if (ports.size() == 1) {
    port = ports.begin()->second;
  } else {
    int id = 0;

    for (it = ports.begin(); it != ports.end(); it++) {
      printf("[%d] %s %s\n", id, it->first.c_str(), it->second.c_str());
      id++;
    }

    if (ports.empty()) {
      printf("Not Lidar was detected. Please enter the lidar serial port:");
      std::cin >> port;
    } else {
      while (ydlidar::os_isOk()) {
        printf("Please select the lidar port:");
        std::string number;
        std::cin >> number;

        if ((size_t)atoi(number.c_str()) >= ports.size()) {
          continue;
        }

        it = ports.begin();
        id = atoi(number.c_str());

        while (id) {
          id--;
          it++;
        }

        port = it->second;
        break;
      }
    }
  }

  int baudrate = 230400;
  bool isSingleChannel = false;
  float frequency = 10.0;

  CYdLidar laser;

  laser.setlidaropt(LidarPropSerialPort, port.c_str(), port.size());

  std::string ignore_array;
  ignore_array.clear();
  laser.setlidaropt(LidarPropIgnoreArray, ignore_array.c_str(), ignore_array.size());

  laser.setlidaropt(LidarPropSerialBaudrate, &baudrate, sizeof(int));

  int optval = TYPE_TRIANGLE;
  laser.setlidaropt(LidarPropLidarType, &optval, sizeof(int));

  optval = YDLIDAR_TYPE_SERIAL;
  laser.setlidaropt(LidarPropDeviceType, &optval, sizeof(int));

  optval = isSingleChannel ? 3 : 4;
  laser.setlidaropt(LidarPropSampleRate, &optval, sizeof(int));

  optval = 4;
  laser.setlidaropt(LidarPropAbnormalCheckCount, &optval, sizeof(int));

  optval = 8;
  laser.setlidaropt(LidarPropIntenstiyBit, &optval, sizeof(int));

  bool b_optvalue = true;
  laser.setlidaropt(LidarPropFixedResolution, &b_optvalue, sizeof(bool));

  b_optvalue = false;
  laser.setlidaropt(LidarPropReversion, &b_optvalue, sizeof(bool));
  laser.setlidaropt(LidarPropInverted, &b_optvalue, sizeof(bool));

  b_optvalue = true;
  laser.setlidaropt(LidarPropAutoReconnect, &b_optvalue, sizeof(bool));

  laser.setlidaropt(LidarPropSingleChannel, &isSingleChannel, sizeof(bool));

  b_optvalue = true;
  laser.setlidaropt(LidarPropIntenstiy, &b_optvalue, sizeof(bool));

  b_optvalue = false;
  laser.setlidaropt(LidarPropSupportMotorDtrCtrl, &b_optvalue, sizeof(bool));
  laser.setlidaropt(LidarPropSupportHeartBeat, &b_optvalue, sizeof(bool));

  float f_optvalue = 180.0f;
  laser.setlidaropt(LidarPropMaxAngle, &f_optvalue, sizeof(float));

  f_optvalue = -180.0f;
  laser.setlidaropt(LidarPropMinAngle, &f_optvalue, sizeof(float));

  f_optvalue = 64.f;
  laser.setlidaropt(LidarPropMaxRange, &f_optvalue, sizeof(float));

  f_optvalue = 0.05f;
  laser.setlidaropt(LidarPropMinRange, &f_optvalue, sizeof(float));

  laser.setlidaropt(LidarPropScanFrequency, &frequency, sizeof(float));

  laser.enableGlassNoise(false);
  laser.enableSunNoise(false);

  bool ret = laser.initialize();
  if (!ret) {
    error("Fail to initialize %s", laser.DescribeError());
    return -1;
  }

  ret = laser.turnOn();
  if (!ret) {
    error("Fail to start %s", laser.DescribeError());
    return -1;
  }

  LaserScan scan;

  while (ydlidar::os_isOk())
  {
    if (laser.doProcessSimple(scan))
    {
      for (size_t i = 0; i < scan.points.size(); ++i)
      {
        const LaserPoint &p = scan.points.at(i);

        float angle = p.angle * 180.0 / M_PI;
        float distance_cm = p.range * 100;  // 🔥 cm

        // filtro de ruido
        if (distance_cm <= 0 || distance_cm > 600) continue;

        printf("%.2f,%.2f\n", angle, distance_cm);
      }

      // ralentizar lectura
      usleep(100000); // 100 ms
    }
    else
    {
      error("Failed to get Lidar Data");
    }
  }

  laser.turnOff();
  laser.disconnecting();

  return 0;
}
