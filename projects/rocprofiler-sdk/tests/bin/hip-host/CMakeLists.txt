#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-bin-hip-host LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(hip-host)
target_sources(hip-host PRIVATE hip_host.cpp)
target_link_libraries(hip-host PRIVATE rocprofiler-sdk::tests-build-flags hip::host
                                       Threads::Threads)
