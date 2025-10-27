# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

if(NOT OMP_TARGET_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH}
        ENV ROCM_PATH
        /opt/rocm
        PATHS ${ROCM_PATH}
        ENV ROCM_PATH
        /opt/rocm
        PATH_SUFFIXES bin llvm/bin
    )
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(OMP_TARGET_COMPILER
            "${amdclangpp_EXECUTABLE}"
            CACHE FILEPATH
            "OpenMP target compiler"
        )
    else()
        message(WARNING "OpenMP target compiler not found. Skipping this example.")
        return()
    endif()
endif()

project(rocprofiler-systems-example-openmp-target-lib LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

find_package(Threads REQUIRED)

function(add_offload_flags tgt)
    foreach(arch IN LISTS GPU_TARGETS)
        target_compile_options(${tgt} PRIVATE --offload-arch=${arch})
        target_link_options(${tgt} PUBLIC --offload-arch=${arch})
    endforeach()
endfunction()

# Derive the ROCm root
get_filename_component(_omp_bin_dir "${OMP_TARGET_COMPILER}" DIRECTORY)
get_filename_component(ROCM_ROOT_DIR "${_omp_bin_dir}" DIRECTORY)

message(STATUS "Using OpenMP target compiler: ${OMP_TARGET_COMPILER}")
message(STATUS "ROCm root inferred from compiler: ${ROCM_ROOT_DIR}")

# Candidate lib directories for libomptarget across ROCm layouts
set(_LLVM_LIB_HINTS
    "${ROCmVersion_DIR}/llvm/lib"
    "${ROCmVersion_DIR}/lib"
    "${ROCM_ROOT_DIR}/lib"
    "${ROCM_ROOT_DIR}/llvm/lib"
    "$ENV{ROCM_PATH}/llvm/lib"
    "$ENV{ROCM_PATH}/lib/llvm/lib"
    "/opt/rocm/llvm/lib"
    "/opt/rocm/lib/llvm/lib"
)

# Find libomptarget
find_library(LIBOMPTARGET_SO NAMES omptarget HINTS ${_LLVM_LIB_HINTS})

if(NOT LIBOMPTARGET_SO)
    message(FATAL_ERROR "Could not find libomptarget in any of:\n  ${_LLVM_LIB_HINTS}")
endif()

# Use the directory that actually contains the library we found
get_filename_component(_rocm_llvm_lib "${LIBOMPTARGET_SO}" DIRECTORY)
set(_rocm_clang_lib "${ROCM_ROOT_DIR}/lib")
set(_COMMON_RPATH "${_rocm_llvm_lib};${_rocm_clang_lib};$ORIGIN")
if(ROCmVersion_DIR)
    list(APPEND _COMMON_RPATH "${ROCmVersion_DIR}/llvm/lib")
endif()
list(REMOVE_DUPLICATES _COMMON_RPATH)

message(STATUS "libomptarget found at: ${LIBOMPTARGET_SO}")
message(STATUS "LLVM libdir: ${_rocm_llvm_lib}")
message(STATUS "Clang libdir: ${_rocm_clang_lib}")

# Shared library

add_library(openmp-target-lib SHARED)
target_sources(openmp-target-lib PRIVATE library.cpp)
target_link_libraries(openmp-target-lib PUBLIC Threads::Threads)
target_compile_options(openmp-target-lib PRIVATE -fopenmp -ggdb)
target_link_options(openmp-target-lib PUBLIC -fopenmp)
add_offload_flags(openmp-target-lib)

# Executable

add_executable(openmp-target)
target_sources(openmp-target PRIVATE main.cpp)
target_link_libraries(openmp-target PRIVATE openmp-target-lib)
target_compile_options(openmp-target PRIVATE -fopenmp -ggdb)
target_link_options(openmp-target PUBLIC -fopenmp)
add_offload_flags(openmp-target)

foreach(tgt openmp-target-lib openmp-target)
    set_target_properties(
        ${tgt}
        PROPERTIES
            BUILD_RPATH "${_COMMON_RPATH}"
            INSTALL_RPATH "${_COMMON_RPATH}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
            POSITION_INDEPENDENT_CODE ON
    )
    target_link_options(
        ${tgt}
        PUBLIC
            "-Wl,--enable-new-dtags"
            "-L${_rocm_llvm_lib}"
            "-Wl,-rpath,${_rocm_llvm_lib}"
            "-Wl,-rpath,${_rocm_clang_lib}"
    )
endforeach()

rocprofiler_systems_custom_compilation(
    TARGET   openmp-target-lib
    COMPILER ${OMP_TARGET_COMPILER}
)

rocprofiler_systems_custom_compilation(TARGET openmp-target
                                       COMPILER ${OMP_TARGET_COMPILER}
)
