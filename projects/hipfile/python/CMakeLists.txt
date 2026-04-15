cmake_minimum_required(VERSION 3.21)
project(hipfile-python LANGUAGES C)

# ---- Python & Cython -----------------------------------------------------

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_program(CYTHON_EXECUTABLE cython REQUIRED)

# ---- hipFile library & headers --------------------------------------------

# hipfile.h location (default: sibling include/ directory)
set(HIPFILE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    CACHE PATH "Path to directory containing hipfile.h")

# libhipfile.so location
find_library(HIPFILE_LIBRARY hipfile
    HINTS
        "${CMAKE_CURRENT_SOURCE_DIR}/../build/src/amd_detail"
        "/opt/rocm/lib"
)
if(NOT HIPFILE_LIBRARY)
    message(FATAL_ERROR
        "Could not find libhipfile. Set -DHIPFILE_LIBRARY=<path> or "
        "install hipfile to /opt/rocm.")
endif()

# HIP runtime headers (needed because hipfile.h includes hip/hip_runtime_api.h)
find_path(HIP_INCLUDE_DIR hip/hip_runtime_api.h
    HINTS
        "/opt/rocm/include"
        "/opt/rocm/hip/include"
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
