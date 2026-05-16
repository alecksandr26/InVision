cmake_minimum_required(VERSION 3.15)
project(lidar_invision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ────────────────────────────────────────────────────────────
find_package(Python  REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# YDLIDAR SDK — installed to /usr/local by default (see README)
# CYdLidar.h is bundled in src/ so we only need the compiled library.
find_library(YDLIDAR_LIB ydlidar_sdk
  HINTS /usr/local/lib /usr/lib
  REQUIRED)

message(STATUS "YDLIDAR SDK lib: ${YDLIDAR_LIB}")

# ── Module ──────────────────────────────────────────────────────────────────
pybind11_add_module(lidar_invision src/binding.cpp)

target_include_directories(lidar_invision PRIVATE
  src                          # lidar_manager.hpp + bundled CYdLidar.h
  /usr/local/include           # YDLIDAR SDK transitive headers (ydlidar_def.h etc.)
  /usr/local/include/ydlidar
)

target_link_libraries(lidar_invision PRIVATE
  ${YDLIDAR_LIB}
  Threads::Threads
)

# ── Platform flags ──────────────────────────────────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
  # Raspberry Pi 5
  target_compile_options(lidar_invision PRIVATE
    -Wall -O3 -march=armv8.2-a+dotprod)
else()
  target_compile_options(lidar_invision PRIVATE
    -Wall -O3 -march=native)
endif()

# Bake rpath so the .so finds ydlidar_sdk at runtime without LD_LIBRARY_PATH
set_target_properties(lidar_invision PROPERTIES
  SKIP_BUILD_RPATH         FALSE
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH            "/usr/local/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)
