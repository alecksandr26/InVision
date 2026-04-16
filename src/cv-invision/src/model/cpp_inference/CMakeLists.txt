cmake_minimum_required(VERSION 3.15)
project(inference_engine_cpp)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# Find the system-installed TensorFlow Lite library
find_library(TENSORFLOW_LITE_LIB
    NAMES tensorflow-lite
    PATHS /usr/lib /usr/local/lib
    REQUIRED
)

# Find the TensorFlow Lite headers (installed to /usr/include/tensorflow)
find_path(TENSORFLOW_LITE_INCLUDE_DIR
    NAMES tensorflow/lite/model.h
    PATHS /usr/include /usr/local/include
    REQUIRED
)

pybind11_add_module(inference_engine_cpp inference_engine.cpp)

target_link_libraries(inference_engine_cpp PRIVATE
    ${TENSORFLOW_LITE_LIB}
    pthread
    dl
)

target_include_directories(inference_engine_cpp PRIVATE
    ${TENSORFLOW_LITE_INCLUDE_DIR}
)

target_compile_features(inference_engine_cpp PRIVATE cxx_std_17)

# Optional: set C++ standard and warnings
target_compile_options(inference_engine_cpp PRIVATE -Wall -Wextra)
