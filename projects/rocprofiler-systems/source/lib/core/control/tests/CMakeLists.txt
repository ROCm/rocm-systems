# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(control-posix-clock-tests OBJECT test_posix_clock.cpp)

target_link_libraries(
    control-posix-clock-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-common-library
        rocprofiler-systems-logger
)

target_include_directories(
    control-posix-clock-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib
)

add_library(control-session-scope-tests OBJECT test_session_scope.cpp)

target_link_libraries(
    control-session-scope-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-common-library
        rocprofiler-systems-logger
)

target_include_directories(
    control-session-scope-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib
)
