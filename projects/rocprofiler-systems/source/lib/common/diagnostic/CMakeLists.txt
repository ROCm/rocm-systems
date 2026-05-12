# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems common::diagnostic
#
#   - color.{hpp,cpp}            ANSI helpers + TTY/NO_COLOR detection
#   - stacktrace.{hpp,cpp}       libunwind capture + libdw symbolize + format
#   - format_exception.{hpp,cpp} formats any std::exception with a trace
#
# Lower layer than core/binary so other internal libraries can depend on it.
#
# ------------------------------------------------------------------------------#

set(diagnostic_sources
    ${CMAKE_CURRENT_LIST_DIR}/color.cpp
    ${CMAKE_CURRENT_LIST_DIR}/stacktrace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/format_exception.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exception.cpp
)

set(diagnostic_headers
    ${CMAKE_CURRENT_LIST_DIR}/color.hpp
    ${CMAKE_CURRENT_LIST_DIR}/stacktrace.hpp
    ${CMAKE_CURRENT_LIST_DIR}/format_exception.hpp
    ${CMAKE_CURRENT_LIST_DIR}/exception.hpp
)

add_library(rocprofiler-systems-diagnostic-library STATIC)
add_library(
    rocprofiler-systems::rocprofiler-systems-diagnostic
    ALIAS rocprofiler-systems-diagnostic-library
)

target_sources(
    rocprofiler-systems-diagnostic-library
    PRIVATE ${diagnostic_sources} ${diagnostic_headers}
)

# The interface library transitively wires up: spdlog (for fmt), libunwind
# (timemory's wrapper), elfutils (libdw + libelf), pthreads, libdl. We
# only depend on the global interface library so the include paths and
# link list match what binary/core use.
target_link_libraries(
    rocprofiler-systems-diagnostic-library
    PUBLIC rocprofiler-systems-common-library
    PRIVATE rocprofiler-systems::rocprofiler-systems-interface-library ${CMAKE_DL_LIBS}
)

set_target_properties(
    rocprofiler-systems-diagnostic-library
    PROPERTIES OUTPUT_NAME ${BINARY_NAME_PREFIX}-diagnostic POSITION_INDEPENDENT_CODE ON
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
