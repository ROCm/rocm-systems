#
#
#
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(CMAKE_CXX_FLAGS_INIT "-W -Wall -Wextra")

project(rocprofiler-register-library-implementation LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(rocprofiler-register REQUIRED)

add_library(rocp-reg-hip-demo SHARED)
target_sources(rocp-reg-hip-demo PRIVATE demo.cpp demo.hpp)
target_link_libraries(rocp-reg-hip-demo
                      PRIVATE rocprofiler-register::rocprofiler-register)
