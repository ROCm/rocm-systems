cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-parallel-overhead LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")
find_package(Threads REQUIRED)
add_executable(parallel-overhead parallel-overhead.cpp)
target_link_libraries(parallel-overhead Threads::Threads)

add_executable(parallel-overhead-locks parallel-overhead.cpp)
target_link_libraries(parallel-overhead-locks Threads::Threads)
target_compile_definitions(parallel-overhead-locks PRIVATE USE_LOCKS=1)

if(NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_target_properties(parallel-overhead PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                       ${CMAKE_BINARY_DIR})
    set_target_properties(parallel-overhead-locks PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                             ${CMAKE_BINARY_DIR})
endif()
