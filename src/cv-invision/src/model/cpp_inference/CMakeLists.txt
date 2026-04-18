# Adding the flags for compiling:
# export CXXFLAGS="-I/usr/include/opencv4 -fopenmp"
# export LDFLAGS="-lopencv_core -lopencv_imgproc -lgomp"

cmake_minimum_required(VERSION 3.15)
project(cpp_inference)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# 1. Force CMake to find YOUR custom build first
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# Message for debugging - check your terminal when running cmake
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

pybind11_add_module(cpp_inference binding.cpp postprocess.cpp preprocess.cpp)

# 2. RPATH is the secret sauce: it bakes the path into the .so file
set_target_properties(cpp_inference PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "/usr/local/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

target_include_directories(cpp_inference PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cpp_inference PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

target_compile_options(cpp_inference PRIVATE -Wall -O3 -march=armv8.2-a+dotprod -fopenmp)
target_compile_features(cpp_inference PRIVATE cxx_std_17)
