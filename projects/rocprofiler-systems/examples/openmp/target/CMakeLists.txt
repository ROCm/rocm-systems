# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

check_rocprofsys_disable_examples("openmp-target" _OPENMP_TARGET_DISABLED)
if(_OPENMP_TARGET_DISABLED)
    return()
endif()

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

if(NOT ROCPROFSYS_GFX_TARGETS)
    rocprofiler_systems_message(WARNING "No GPU targets detected/set. Disabling OpenMP target example...")
    return()
endif()

if(NOT OMP_TARGET_COMPILER)
    # Prefer amdclang++ from the same bin/ directory as the active CXX compiler
    # so its LLVM version matches the ASan bitcode (asanrtl.bc) built into the
    # same toolchain tree. Use if(EXISTS) rather than find_program to bypass
    # the result cache — a stale entry from a different ROCm installation would
    # otherwise take precedence over the path derived from the active compiler.
    get_filename_component(_cxx_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_toolchain_amdclangpp "${_cxx_bin_dir}/amdclang++")

    if(EXISTS "${_toolchain_amdclangpp}")
        set(OMP_TARGET_COMPILER
            "${_toolchain_amdclangpp}"
            CACHE FILEPATH
            "OpenMP target compiler"
        )
    else()
        # Fall back to the amdclang++ discovered by openmp-helper.cmake
        set(OMP_TARGET_COMPILER
            "${amdclangpp_EXECUTABLE}"
            CACHE FILEPATH
            "OpenMP target compiler"
        )
    endif()
endif()

project(rocprofiler-systems-example-openmp-target-lib LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

find_package(Threads REQUIRED)

function(add_offload_flags tgt)
    foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
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
set(_COMMON_RPATH "${_rocm_llvm_lib};${_rocm_clang_lib};$ORIGIN;$ORIGIN/lib")
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

# Only instinct GPUs are xnack-capable and ship a device-side AddressSanitizer runtime.
# When building under a sanitizer on non-instinct arches, the OpenMP target offload
# device image would otherwise be instrumented and fail to link with undefined
# __asan_* device symbols, so we disable device-side instrumentation with
# -fno-gpu-sanitize. Note: rocprofiler_systems_lookup_gfx defaults unknown arches to
# 'instinct', so device instrumentation is only disabled when every target is
# known-non-instinct (fail-safe).

# NOTE: only inspects CMAKE_CXX_FLAGS. Sanitizers added via add_compile_options,
# CMAKE_CXX_FLAGS_<CONFIG>, or a toolchain file will not be detected; pass
# -fsanitize=... in CMAKE_CXX_FLAGS when building this example under a sanitizer.
string(FIND "${CMAKE_CXX_FLAGS}" "-fsanitize" _omp_target_sanitizer_pos)
if(_omp_target_sanitizer_pos GREATER -1)
    set(_omp_target_device_sanitizer OFF)
    foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
        rocprofiler_systems_lookup_gfx(${arch} _gpu_categories)
        if("instinct" IN_LIST _gpu_categories)
            set(_omp_target_device_sanitizer ON)
            break()
        endif()
    endforeach()
    if(NOT _omp_target_device_sanitizer)
        foreach(tgt openmp-target-lib openmp-target)
            target_compile_options(${tgt} PRIVATE -fno-gpu-sanitize)
            target_link_options(${tgt} PUBLIC -fno-gpu-sanitize)
        endforeach()
    endif()
endif()

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
    TARGET openmp-target-lib
    COMPILER ${OMP_TARGET_COMPILER}
)

rocprofiler_systems_custom_compilation(TARGET openmp-target
    COMPILER ${OMP_TARGET_COMPILER}
)

if(ROCPROFSYS_INSTALL_EXAMPLES)
    if(TARGET openmp-target AND TARGET openmp-target-lib)
        install(
            TARGETS openmp-target openmp-target-lib
            RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
            LIBRARY
                DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples/lib
                COMPONENT rocprofiler-systems-examples
        )
    endif()
endif()
