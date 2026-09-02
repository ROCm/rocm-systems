# RCCL proxy-trace profiler plugin (librccl-profiler-proxytrace.so)
#
# Can be built two ways:
#   1. Standalone:   cmake -S . -B build && cmake --build build
#   2. As part of the main RCCL build via add_subdirectory(plugins/profiler/proxytrace)
#
# Both builds use the local proxy_trace.{h,cc} copies; no dependency on the
# RCCL CMake build directory or external fmt.

# Detect standalone vs. in-tree configuration.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(rccl-profiler-proxytrace LANGUAGES CXX)
    set(PROXYTRACE_STANDALONE ON)
else()
    set(PROXYTRACE_STANDALONE OFF)
endif()

add_library(rccl-profiler-proxytrace SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/proxytrace_plugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/proxy_trace.cc
)

target_include_directories(rccl-profiler-proxytrace PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl
    ${ROCM_PATH}/include
)

target_compile_definitions(rccl-profiler-proxytrace PRIVATE
    RCCL_PROXYTRACE_PLUGIN_BUILD
    NCCL_OS_LINUX
    __HIP_PLATFORM_AMD__
)

target_compile_features(rccl-profiler-proxytrace PRIVATE cxx_std_17)

target_compile_options(rccl-profiler-proxytrace PRIVATE -Wall -Wextra)

# Match the legacy Makefile output name: librccl-profiler-proxytrace.so
set_target_properties(rccl-profiler-proxytrace PROPERTIES
    OUTPUT_NAME "rccl-profiler-proxytrace"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
)

if(NOT PROXYTRACE_STANDALONE)
    # ncclProfilerProxyTraceSiblingPath() looks for this plugin beside the library.
    set_target_properties(rccl-profiler-proxytrace PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
