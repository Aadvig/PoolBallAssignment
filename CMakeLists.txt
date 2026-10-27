
cmake_minimum_required(VERSION 3.16)
project(PoolBallAssignment)

set(CMAKE_CXX_STANDARD 17)

# Use OpenCV 4 instead of OpenCV 5
set(OpenCV_DIR "/opt/homebrew/opt/opencv@4/lib/cmake/opencv4")

find_package(OpenCV 4 REQUIRED)

add_executable(PoolBallAssignment main.cpp)

target_include_directories(PoolBallAssignment PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(PoolBallAssignment PRIVATE ${OpenCV_LIBS})
