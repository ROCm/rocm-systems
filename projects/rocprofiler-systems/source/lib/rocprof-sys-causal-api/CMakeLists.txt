# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems causal API: minimal library backing ROCPROFSYS_CAUSAL_*;
# linked directly by apps, with librocprof-sys-dl.so providing the real callbacks.
#
# ------------------------------------------------------------------------------#

set(CMAKE_SKIP_RPATH OFF)
set(BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(rocprofiler-systems-causal-api-library SHARED)
add_library(
    rocprofiler-systems::rocprofiler-systems-causal-api-library
    ALIAS rocprofiler-systems-causal-api-library
)

set(_causal_api_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-systems/causal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-systems/causal_api.h
)

set(_causal_api_sources ${CMAKE_CURRENT_SOURCE_DIR}/causal_api.cpp)

target_sources(
    rocprofiler-systems-causal-api-library
    PRIVATE ${_causal_api_sources} ${_causal_api_headers}
)
target_include_directories(
    rocprofiler-systems-causal-api-library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
    rocprofiler-systems-causal-api-library
    PUBLIC rocprofiler-systems::rocprofiler-systems-common-api-library
    PRIVATE
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-compile-definitions>
)

set_target_properties(
    rocprofiler-systems-causal-api-library
    PROPERTIES
        OUTPUT_NAME ${BINARY_NAME_PREFIX}-causal-api
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        POSITION_INDEPENDENT_CODE ON
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
)

rocprofiler_systems_strip_target(rocprofiler-systems-causal-api-library)

rocprofiler_systems_install_public_headers(${_causal_api_headers})

install(
    TARGETS rocprofiler-systems-causal-api-library
    EXPORT rocprofiler-systems-library-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
