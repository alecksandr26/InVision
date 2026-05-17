cmake_minimum_required(VERSION 3.15)
project(lidar_invision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ────────────────────────────────────────────────────────────
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# YDLIDAR SDK — installed to /usr/local by default
find_library(YDLIDAR_LIB ydlidar_sdk
    HINTS /usr/local/lib /usr/lib
    REQUIRED)

message(STATUS "YDLIDAR SDK lib: ${YDLIDAR_LIB}")

# ── Shared Include Directories ──────────────────────────────────────────────
# We pull this out so both the Python module and C++ binary share the same paths
set(LIDAR_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src  # lidar_manager.hpp + bundled CYdLidar.h
    /usr/local/include               # YDLIDAR SDK transitive headers
    /usr/local/include/ydlidar
    /usr/local/include/src           # Core SDK utility headers
)

# ── Target 1: Python Binding Module (.so) ───────────────────────────────────
pybind11_add_module(lidar_invision src/binding.cpp)
target_include_directories(lidar_invision PRIVATE ${LIDAR_INCLUDE_DIRS})
target_link_libraries(lidar_invision PRIVATE ${YDLIDAR_LIB} Threads::Threads)

# ── Target 2: Native C++ Test Executable (lidar_test_cpp) ───────────────────
add_executable(lidar_test_cpp src/test.cpp)
target_include_directories(lidar_test_cpp PRIVATE ${LIDAR_INCLUDE_DIRS})
target_link_libraries(lidar_test_cpp PRIVATE ${YDLIDAR_LIB} Threads::Threads)

# ── Platform Compiler Flags ──────────────────────────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    # Optimized build flags targeting Raspberry Pi hardware architectures
    set(PLATFORM_FLAGS "-Wall" "-O3" "-march=armv8.2-a+dotprod")
else()
    set(PLATFORM_FLAGS "-Wall" "-O3" "-march=native")
endif()

target_compile_options(lidar_invision PRIVATE ${PLATFORM_FLAGS})
target_compile_options(lidar_test_cpp PRIVATE ${PLATFORM_FLAGS})

# Bake rpath so both targets find ydlidar_sdk at runtime without LD_LIBRARY_PATH
set_target_properties(lidar_invision lidar_test_cpp PROPERTIES
    SKIP_BUILD_RPATH            FALSE
    BUILD_WITH_INSTALL_RPATH    TRUE
    INSTALL_RPATH               "/usr/local/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
