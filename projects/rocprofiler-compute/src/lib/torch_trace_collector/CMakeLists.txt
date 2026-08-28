## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(BUILD_TORCH_TRACE_COLLECTOR
    "AUTO"
    CACHE STRING
    "Build torch_trace_collector (PyTorch 2.13 or 2.14): AUTO, ON, or OFF"
)
set_property(CACHE BUILD_TORCH_TRACE_COLLECTOR PROPERTY STRINGS AUTO ON OFF)

if(NOT BUILD_TORCH_TRACE_COLLECTOR MATCHES "^(AUTO|ON|OFF)$")
    message(
        FATAL_ERROR
        "BUILD_TORCH_TRACE_COLLECTOR must be AUTO, ON or OFF (got "
        "'${BUILD_TORCH_TRACE_COLLECTOR}')."
    )
endif()

if(BUILD_TORCH_TRACE_COLLECTOR STREQUAL "OFF")
    return()
endif()

macro(_skip_collector reason)
    if(BUILD_TORCH_TRACE_COLLECTOR STREQUAL "ON")
        message(FATAL_ERROR "torch_trace_collector: ${reason}")
    else()
        message(STATUS "torch_trace_collector: ${reason}")
    endif()
    return()
endmacro()

if(NOT DEFINED ENV{ROCM_PATH} OR "$ENV{ROCM_PATH}" STREQUAL "")
    _skip_collector("ROCM_PATH is not set")
endif()
get_filename_component(_rocm_path "$ENV{ROCM_PATH}" ABSOLUTE)
get_filename_component(_torch_prefix "${_rocm_path}/../torch" ABSOLUTE)
find_package(Torch CONFIG QUIET PATHS "${_torch_prefix}" NO_DEFAULT_PATH)
if(NOT TORCH_FOUND)
    _skip_collector("find_package(Torch) failed at ${_torch_prefix}")
endif()

set(_torch_version "${Torch_VERSION}")
string(REGEX REPLACE "\\+.*$" "" _torch_version "${_torch_version}")
if(NOT _torch_version MATCHES "^2\\.(13|14)([.]|$)")
    _skip_collector(
        "Torch ${_torch_version} is not supported; requires 2.13 or 2.14"
    )
endif()

set(_module_target "torch_trace_collector-${_torch_version}")
set(_torch_lib_dir "${TORCH_INSTALL_PREFIX}/lib")

find_library(
    ROCTX_LIBRARY
    NAMES rocprofiler-sdk-roctx
    HINTS ${_rocm_path}
    PATH_SUFFIXES lib lib64
)
if(NOT ROCTX_LIBRARY)
    _skip_collector("rocprofiler-sdk-roctx library not found")
endif()
find_path(
    ROCTX_INCLUDE_DIR
    NAMES rocprofiler-sdk-roctx/roctx.h
    HINTS ${_rocm_path}
    PATH_SUFFIXES include
)
if(NOT ROCTX_INCLUDE_DIR)
    _skip_collector("rocprofiler-sdk-roctx headers not found")
endif()

find_package(Python3 COMPONENTS Development.Module QUIET)
if(NOT Python3_Development.Module_FOUND)
    _skip_collector("Python 3 development headers not found")
endif()

add_library(torch-trace-collector STATIC torch_trace_collector.cpp snapshot_store.cpp)
target_include_directories(
    torch-trace-collector
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${TORCH_INCLUDE_DIRS} ${ROCTX_INCLUDE_DIR}
)
target_link_libraries(
    torch-trace-collector
    PUBLIC ${TORCH_LIBRARIES} ${ROCTX_LIBRARY} synchronized gsl_assert
)
target_compile_options(
    torch-trace-collector
    PRIVATE -fvisibility=hidden -Wno-deprecated-declarations
)

add_library(${_module_target} MODULE torch_trace_collector_module.cpp)
target_link_libraries(${_module_target} PRIVATE torch-trace-collector Python3::Module)
target_compile_options(
    ${_module_target}
    PRIVATE -fvisibility=hidden -Wno-deprecated-declarations
)
set_target_properties(${_module_target} PROPERTIES PREFIX "")

set(TORCH_TRACE_COLLECTOR_TARGET "${_module_target}" CACHE INTERNAL "")

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

message(STATUS "torch_trace_collector: ${_module_target} (${TORCH_INSTALL_PREFIX})")
