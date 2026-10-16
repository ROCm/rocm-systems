# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(clocks-thread-cpu-tests OBJECT test_thread_cpu.cpp)

target_link_libraries(
    clocks-thread-cpu-tests
    PRIVATE rocprofiler-systems-googletest-library
)

target_include_directories(
    clocks-thread-cpu-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib
)

add_library(clocks-timeline-tests OBJECT test_timeline.cpp)

target_link_libraries(
    clocks-timeline-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-common-library
        rocprofiler-systems-logger
)

target_include_directories(
    clocks-timeline-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib
)

add_library(clocks-waitable-posix-tests OBJECT test_waitable_posix.cpp)

target_link_libraries(
    clocks-waitable-posix-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-common-library
        rocprofiler-systems-logger
)

target_include_directories(
    clocks-waitable-posix-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib
)

add_library(clocks-waitable-steady-tests OBJECT test_waitable_steady.cpp)

target_link_libraries(
    clocks-waitable-steady-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-common-library
        rocprofiler-systems-logger
)

target_include_directories(
    clocks-waitable-steady-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib
)
