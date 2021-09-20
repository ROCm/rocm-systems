cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(hosttrace-parallel-overhead LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")
find_package(Threads REQUIRED)
add_executable(parallel-overhead parallel-overhead.cpp)
target_link_libraries(parallel-overhead Threads::Threads)

if(NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_target_properties(parallel-overhead PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                       ${CMAKE_BINARY_DIR})
endif()
