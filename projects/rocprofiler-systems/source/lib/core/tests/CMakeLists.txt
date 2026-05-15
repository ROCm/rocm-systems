# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(demangler-tests OBJECT demangler_test.cpp)

target_link_libraries(
    demangler-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(demangler-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(track-registry-tests OBJECT test_track_registry.cpp)

target_link_libraries(
    track-registry-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(
    track-registry-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib
)
