## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

# Builds the collector against the PyTorch install beside ROCM_PATH.
macro(_skip_collector reason)
    message(WARNING "torch_trace_collector not built: ${reason}")
    return()
endmacro()

if(NOT DEFINED ENV{ROCM_PATH} OR "$ENV{ROCM_PATH}" STREQUAL "")
    _skip_collector("ROCM_PATH is not set")
endif()
get_filename_component(_rocm_path "$ENV{ROCM_PATH}" ABSOLUTE)
get_filename_component(_torch_prefix "${_rocm_path}/../torch" ABSOLUTE)

set(_torch_version_headers
    "${_torch_prefix}/include/torch/headeronly/version.h"
    "${_torch_prefix}/include/torch/csrc/api/include/torch/version.h"
)
set(_torch_version "")
foreach(_header IN LISTS _torch_version_headers)
    if(NOT EXISTS "${_header}")
        continue()
    endif()
    file(STRINGS "${_header}" _major_line REGEX "define[ \t]+TORCH_VERSION_MAJOR")
    file(STRINGS "${_header}" _minor_line REGEX "define[ \t]+TORCH_VERSION_MINOR")
    if(_major_line AND _minor_line)
        string(REGEX REPLACE "[^0-9]" "" _torch_major "${_major_line}")
        string(REGEX REPLACE "[^0-9]" "" _torch_minor "${_minor_line}")
        set(_torch_version "${_torch_major}.${_torch_minor}")
        break()
    endif()
endforeach()

if(NOT _torch_version)
    _skip_collector("no PyTorch version header under ${_torch_prefix}")
endif()
if(NOT _torch_version MATCHES "^2\\.(13|14)$")
    _skip_collector(
        "PyTorch ${_torch_version} is not supported; requires 2.13 or 2.14"
    )
endif()

set(_torch_lib_dir "${_torch_prefix}/lib")
set(_torch_include_dirs
    "${_torch_prefix}/include"
    "${_torch_prefix}/include/torch/csrc/api/include"
)
foreach(_lib IN ITEMS torch torch_cpu c10)
    find_library(
        TORCH_${_lib}_LIBRARY
        NAMES ${_lib}
        PATHS "${_torch_lib_dir}"
        NO_DEFAULT_PATH
    )
    if(NOT TORCH_${_lib}_LIBRARY)
        _skip_collector("lib${_lib} not found in ${_torch_lib_dir}")
    endif()
    list(APPEND _torch_libraries "${TORCH_${_lib}_LIBRARY}")
endforeach()

find_package(rocprofiler-sdk-roctx CONFIG)
if(NOT rocprofiler-sdk-roctx_FOUND)
    _skip_collector("rocprofiler-sdk-roctx not found")
endif()

# Headers only; the interpreter that loads the module resolves Python symbols.
find_package(Python3 COMPONENTS Development.Module QUIET)
if(NOT Python3_Development.Module_FOUND)
    _skip_collector("Python 3 development headers not found")
endif()

# Skip rather than let target_compile_features fail the whole configure.
if(NOT cxx_std_20 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    _skip_collector("compiler does not support C++20")
endif()

add_library(
    torch-trace-collector
    STATIC
    torch_trace_collector.cpp
    snapshot_store.cpp
    schema_arg_names.cpp
)
target_include_directories(torch-trace-collector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(torch-trace-collector SYSTEM PUBLIC ${_torch_include_dirs})
target_link_libraries(
    torch-trace-collector
    PUBLIC
        ${_torch_libraries}
        rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library
        synchronized
        gsl_assert
)
# PyTorch headers need a newer standard than the C++17 used across src/lib.
target_compile_features(torch-trace-collector PUBLIC cxx_std_20)
target_compile_options(torch-trace-collector PRIVATE -fvisibility=hidden)

add_library(torch-trace-collector-module MODULE torch_trace_collector_module.cpp)
target_link_libraries(torch-trace-collector-module PRIVATE torch-trace-collector)
target_include_directories(
    torch-trace-collector-module
    SYSTEM
    PRIVATE ${Python3_INCLUDE_DIRS}
)
target_compile_options(torch-trace-collector-module PRIVATE -fvisibility=hidden)
set_target_properties(
    torch-trace-collector-module
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "torch_trace_collector-${_torch_version}"
        SUFFIX ".${Python3_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
