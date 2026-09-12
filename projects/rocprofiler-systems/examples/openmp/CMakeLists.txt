# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

if(ROCPROFSYS_DISABLE_EXAMPLES)
    if("openmp" IN_LIST ROCPROFSYS_DISABLE_EXAMPLES)
        return()
    endif()
endif()

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(rocprofiler-systems-openmp LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

file(GLOB common_source ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp)
add_library(openmp-common OBJECT ${common_source})
target_include_directories(openmp-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)

include(${CMAKE_CURRENT_LIST_DIR}/openmp-helper.cmake)

target_compile_options(openmp-common PUBLIC -fopenmp)
target_link_options(openmp-common PUBLIC -fopenmp)
rocprofiler_systems_custom_compilation(
    COMPILER ${amdclangpp_EXECUTABLE}
    TARGET openmp-common
)

add_subdirectory(cg)
add_subdirectory(lu)
# flang cannot lower Fortran OpenMP target-offload to the portable SPIR-V target
# (amdgcnspirv): "Target.cpp: not yet implemented: target not implemented". Skip
# the Fortran examples when the only requested GPU target is amdgcnspirv; native
# targets are unaffected. (The fortran subdir also self-disables offload when no
# GPU target is set.)
set(_rocprofsys_non_spirv_gfx_targets ${ROCPROFSYS_GFX_TARGETS})
list(REMOVE_ITEM _rocprofsys_non_spirv_gfx_targets amdgcnspirv)
if(_rocprofsys_non_spirv_gfx_targets OR NOT ROCPROFSYS_GFX_TARGETS)
    add_subdirectory(fortran)
endif()
add_subdirectory(target)
