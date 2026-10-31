
cmake_minimum_required(VERSION 3.16)
project(PoolBallAssignment)

set(CMAKE_CXX_STANDARD 17)

# =====================================================
# OpenCV 4
# =====================================================

set(OpenCV_DIR "/opt/homebrew/opt/opencv@4/lib/cmake/opencv4")

find_package(OpenCV 4 REQUIRED)

# =====================================================
# XLSXWriter
# =====================================================

find_package(PkgConfig REQUIRED)
pkg_check_modules(XLSXWRITER REQUIRED IMPORTED_TARGET xlsxwriter)

# =====================================================
# Executable
# =====================================================

add_executable(PoolBallAssignment main.cpp)

# =====================================================
# Include directories
# =====================================================

target_include_directories(PoolBallAssignment PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

# =====================================================
# Link libraries
# =====================================================

target_link_libraries(PoolBallAssignment PRIVATE
        ${OpenCV_LIBS}
        PkgConfig::XLSXWRITER
)
