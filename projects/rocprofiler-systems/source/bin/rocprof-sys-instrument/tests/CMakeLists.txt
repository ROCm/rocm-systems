# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocprof-sys-instrument-tests OBJECT test_arg_serializer.cpp)

# Includes:
#   - .. so we can include "arg_serializer.hpp" directly
#   - source/lib so core/demangler.hpp and core/common_types.hpp resolve
target_include_directories(
    rocprof-sys-instrument-tests
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/.. ${PROJECT_SOURCE_DIR}/source/lib
)

target_link_libraries(
    rocprof-sys-instrument-tests
    PUBLIC rocprofiler-systems-googletest-library
)
