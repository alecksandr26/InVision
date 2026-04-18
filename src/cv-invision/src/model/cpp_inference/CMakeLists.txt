cmake_minimum_required(VERSION 3.15)
project(postprocess_cpp)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(postprocess_cpp postprocess.cpp)

target_compile_features(postprocess_cpp PRIVATE cxx_std_17)
target_compile_options(postprocess_cpp PRIVATE -Wall -Wextra -O3 -march=native)

