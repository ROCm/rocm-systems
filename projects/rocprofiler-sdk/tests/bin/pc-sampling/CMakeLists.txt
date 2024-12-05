#
# Integration test applications
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-tests-bin LANGUAGES C CXX)

set(CMAKE_BUILD_RPATH "\$ORIGIN:\$ORIGIN/../lib")

# applications used by integration tests which DO NOT link to rocprofiler-sdk-roctx
add_subdirectory(exec-mask-manipulation)
