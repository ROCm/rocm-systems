# RCCL proxy-trace profiler plugin (librccl-profiler-proxytrace.so)
#
# Can be built two ways:
#   1. Standalone:   cmake -S . -B build && cmake --build build
#   2. As part of the main RCCL build via add_subdirectory(plugins/profiler/proxytrace)
#
# Either way, fmt is resolved through the imported fmt::fmt-header-only target,
# which provides the include path (e.g. an installed/conda fmt) and the
# FMT_HEADER_ONLY definition automatically.

# Detect standalone vs. in-tree configuration.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(rccl-profiler-proxytrace LANGUAGES CXX)
    set(PROXYTRACE_STANDALONE ON)
else()
    set(PROXYTRACE_STANDALONE OFF)
endif()

# RCCL source root (this file lives at plugins/profiler/proxytrace).
get_filename_component(RCCL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# fmt::fmt-header-only is already available when built in-tree (the main build
# calls find_package / FetchContent for fmt). For standalone builds, resolve it.
if(NOT TARGET fmt::fmt-header-only)
    find_package(fmt REQUIRED)
endif()

add_library(rccl-profiler-proxytrace SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/proxytrace_plugin.cc
    ${RCCL_ROOT}/src/misc/proxy_trace/proxy_trace.cc
)

target_include_directories(rccl-profiler-proxytrace PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RCCL_ROOT}/src/include/plugin
    ${RCCL_ROOT}/src/include
)

target_compile_definitions(rccl-profiler-proxytrace PRIVATE
    RCCL_PROXYTRACE_PLUGIN_BUILD
)

target_compile_features(rccl-profiler-proxytrace PRIVATE cxx_std_17)

target_compile_options(rccl-profiler-proxytrace PRIVATE -Wall -Wextra)

# fmt-header-only supplies the fmt include directory and FMT_HEADER_ONLY=1.
target_link_libraries(rccl-profiler-proxytrace PRIVATE fmt::fmt-header-only)

# Match the legacy Makefile output name: librccl-profiler-proxytrace.so
set_target_properties(rccl-profiler-proxytrace PROPERTIES
    OUTPUT_NAME "rccl-profiler-proxytrace"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
)

if(NOT PROXYTRACE_STANDALONE)
    set_target_properties(rccl-profiler-proxytrace PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()
