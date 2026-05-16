cmake_minimum_required(VERSION 3.5)
project(lidar_project)

set(CMAKE_CXX_STANDARD 11)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)


add_executable(tmini_test src/tmini_test.cpp)

target_link_libraries(tmini_test
    ydlidar_sdk
    pthread
)

target_include_directories(tmini_test
        PUBLIC src
)
