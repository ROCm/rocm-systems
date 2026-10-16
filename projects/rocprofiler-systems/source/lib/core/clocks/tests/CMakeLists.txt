# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(clocks-thread-cputime-tests OBJECT test_thread_cputime.cpp)

target_link_libraries(
    clocks-thread-cputime-tests
    PRIVATE rocprofiler-systems-googletest-library
)

target_include_directories(
    clocks-thread-cputime-tests
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
