
cmake_minimum_required(VERSION 3.16)
project(PoolBallAssignment)

set(CMAKE_CXX_STANDARD 17)

# Homebrew OpenCV installation
set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/opencv")

find_package(OpenCV REQUIRED)

add_executable(PoolBallAssignment main.cpp)

target_include_directories(PoolBallAssignment PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(PoolBallAssignment PRIVATE ${OpenCV_LIBS})
