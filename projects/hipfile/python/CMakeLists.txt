cmake_minimum_required(VERSION 3.21)
project(hipfile-python LANGUAGES C)

# ---- Python & Cython -----------------------------------------------------

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_program(CYTHON_EXECUTABLE cython REQUIRED)

# ---- hipFile library & headers --------------------------------------------

# ROCm install hints: prefer $ROCM_PATH from env; fall back to a glob of
# /opt/rocm/core-* (7.11+ layout) plus legacy /opt/rocm.
if(DEFINED ENV{ROCM_PATH})
    set(_ROCM_ROOTS "$ENV{ROCM_PATH}")
else()
    file(GLOB _ROCM_ROOTS "/opt/rocm/core-*")
    list(APPEND _ROCM_ROOTS "/opt/rocm")
endif()

set(_ROCM_INCLUDE_HINTS)
set(_ROCM_LIB_HINTS)
foreach(_r IN LISTS _ROCM_ROOTS)
    list(APPEND _ROCM_INCLUDE_HINTS "${_r}/include")
    list(APPEND _ROCM_LIB_HINTS "${_r}/lib")
endforeach()

# hipfile.h location (prefer sibling include/ from source, fall back to installed)
find_path(HIPFILE_INCLUDE_DIR hipfile.h
    HINTS
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        ${_ROCM_INCLUDE_HINTS}
    PATH "Path to directory containing hipfile.h"
)
if(NOT HIPFILE_INCLUDE_DIR)
    message(FATAL_ERROR
        "Could not find hipfile.h. Set -DHIPFILE_INCLUDE_DIR=<path> or "
        "install hipfile to /opt/rocm.")
endif()

# libhipfile.so location
find_library(HIPFILE_LIBRARY hipfile
    HINTS
        "${CMAKE_CURRENT_SOURCE_DIR}/../build/src/amd_detail"
        ${_ROCM_LIB_HINTS}
)
if(NOT HIPFILE_LIBRARY)
    message(FATAL_ERROR
        "Could not find libhipfile. Set -DHIPFILE_LIBRARY=<path> or "
        "install hipfile to /opt/rocm.")
endif()

# HIP runtime headers (needed because hipfile.h includes hip/hip_runtime_api.h)
find_path(HIP_INCLUDE_DIR hip/hip_runtime_api.h
    HINTS
        ${_ROCM_INCLUDE_HINTS}
)
if(NOT HIP_INCLUDE_DIR)
    message(FATAL_ERROR
        "Could not find hip/hip_runtime_api.h. Set -DHIP_INCLUDE_DIR=<path> "
        "or install HIP development headers.")
endif()

# ---- Cythonize ------------------------------------------------------------

set(_PYX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hipfile/_hipfile.pyx")
set(_C_OUT   "${CMAKE_CURRENT_BINARY_DIR}/_hipfile.c")

add_custom_command(
    OUTPUT  "${_C_OUT}"
    COMMAND "${CYTHON_EXECUTABLE}" "${_PYX_SRC}" -o "${_C_OUT}" -3
    DEPENDS
        "${_PYX_SRC}"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipfile/_chipfile.pxd"
    COMMENT "Cythonizing _hipfile.pyx"
)

# ---- Build extension module -----------------------------------------------
# lint_cmake: -readability/wonkycase
Python_add_library(_hipfile MODULE "${_C_OUT}" WITH_SOABI)
# lint_cmake: +readability/wonkycase
target_include_directories(_hipfile PRIVATE
    "${HIPFILE_INCLUDE_DIR}"
    "${HIP_INCLUDE_DIR}"
)

target_link_libraries(_hipfile PRIVATE "${HIPFILE_LIBRARY}")

# _hipfile bindings only support AMD for now
target_compile_definitions(_hipfile PRIVATE __HIP_PLATFORM_AMD__)

install(TARGETS _hipfile DESTINATION hipfile)
