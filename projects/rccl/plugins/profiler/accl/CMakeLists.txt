# RCCL accl-profiler plugin (librccl-profiler-accl.so)
#
# Can be built two ways:
#   1. Standalone:   cmake -S . -B build && cmake --build build
#   2. As part of the main RCCL build via add_subdirectory(plugins/profiler/accl)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(rccl-profiler-accl LANGUAGES CXX)
    set(ACCL_STANDALONE ON)
else()
    set(ACCL_STANDALONE OFF)
endif()

# RCCL source root (this file lives at plugins/profiler/accl).
get_filename_component(RCCL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

add_library(rccl-profiler-accl SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/accl_profiler.cc
)

target_include_directories(rccl-profiler-accl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RCCL_ROOT}/src/include/plugin
    ${RCCL_ROOT}/src/include
)

target_compile_features(rccl-profiler-accl PRIVATE cxx_std_14)

target_compile_options(rccl-profiler-accl PRIVATE -Wall -Wextra)

target_link_libraries(rccl-profiler-accl PRIVATE pthread)

set_target_properties(rccl-profiler-accl PROPERTIES
    OUTPUT_NAME "rccl-profiler-accl"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    # Callbacks are __hidden; only ncclProfiler_v5 needs default visibility for dlsym.
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

if(NOT ACCL_STANDALONE)
    set_target_properties(rccl-profiler-accl PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()
