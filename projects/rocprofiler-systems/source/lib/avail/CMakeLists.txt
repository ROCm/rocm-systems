# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocprofiler-systems-avail-library STATIC)
add_library(
    rocprofiler-systems::rocprofiler-systems-avail-library
    ALIAS rocprofiler-systems-avail-library
)

target_sources(
    rocprofiler-systems-avail-library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/catalog.cpp
        ${CMAKE_CURRENT_LIST_DIR}/catalog.hpp
        ${CMAKE_CURRENT_LIST_DIR}/records.cpp
        ${CMAKE_CURRENT_LIST_DIR}/records.hpp
)

target_include_directories(
    rocprofiler-systems-avail-library
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/lib>
)

target_link_libraries(
    rocprofiler-systems-avail-library
    PRIVATE
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-compile-options
)

set_target_properties(
    rocprofiler-systems-avail-library
    PROPERTIES OUTPUT_NAME ${BINARY_NAME_PREFIX}-avail-library
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
