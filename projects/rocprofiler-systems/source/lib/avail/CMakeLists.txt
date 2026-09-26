# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems-avail-library: capability catalog queried in-process by
# `rocsys avail`. Holds no formatting and no CLI state.
#
# ------------------------------------------------------------------------------#

set(avail_sources
    ${CMAKE_CURRENT_LIST_DIR}/records.cpp
    ${CMAKE_CURRENT_LIST_DIR}/devices.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_counters.cpp
    ${CMAKE_CURRENT_LIST_DIR}/traces.cpp
    ${CMAKE_CURRENT_LIST_DIR}/catalog.cpp
)

set(avail_headers
    ${CMAKE_CURRENT_LIST_DIR}/records.hpp
    ${CMAKE_CURRENT_LIST_DIR}/devices.hpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_counters.hpp
    ${CMAKE_CURRENT_LIST_DIR}/traces.hpp
    ${CMAKE_CURRENT_LIST_DIR}/catalog.hpp
)

add_library(rocprofiler-systems-avail-library STATIC)
add_library(
    rocprofiler-systems::rocprofiler-systems-avail
    ALIAS rocprofiler-systems-avail-library
)

target_sources(
    rocprofiler-systems-avail-library
    PRIVATE ${avail_sources} ${avail_headers}
)

target_include_directories(
    rocprofiler-systems-avail-library
    PUBLIC ${PROJECT_SOURCE_DIR}/source/lib
)

target_link_libraries(
    rocprofiler-systems-avail-library
    PRIVATE
        rocprofiler-systems::rocprofiler-systems-interface-library
        rocprofiler-systems::rocprofiler-systems-core
        rocprofiler-systems::rocprofiler-systems-amd-smi-backend
)

# rocprofiler-systems-rocprofiler-sdk-backend only exists for ROCm >= 6.4 (see
# source/lib/backends/CMakeLists.txt). Without it, counters and traces report an
# unavailable backend instead of failing to build.
if(TARGET rocprofiler-systems-rocprofiler-sdk-backend)
    target_link_libraries(
        rocprofiler-systems-avail-library
        PRIVATE rocprofiler-systems::rocprofiler-systems-rocprofiler-sdk-backend
    )
    target_compile_definitions(
        rocprofiler-systems-avail-library
        PRIVATE ROCPROFSYS_AVAIL_SDK_BACKEND=1
    )
endif()

set_target_properties(
    rocprofiler-systems-avail-library
    PROPERTIES
        OUTPUT_NAME ${BINARY_NAME_PREFIX}-avail
        POSITION_INDEPENDENT_CODE ON
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
