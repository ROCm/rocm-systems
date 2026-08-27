## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
##
## Builds the torch-trace-collector library and the
## torch_trace_collector-<torch-version>.so pybind module for --torch-trace.
set(BUILD_TORCH_TRACE_COLLECTOR
    "AUTO"
    CACHE STRING
    "Build the torch_trace_collector extension: AUTO (skip when unavailable), ON (require), OFF"
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

macro(_ttc_skip reason)
    if(BUILD_TORCH_TRACE_COLLECTOR STREQUAL "ON")
        message(
            FATAL_ERROR
            "torch_trace_collector: ${reason}; the extension is not built."
        )
    else()
        message(STATUS "torch_trace_collector: ${reason}; the extension is not built.")
    endif()
    return()
endmacro()

# Pin the interpreter used to find Torch and compile the pybind module.
if(TORCH_TRACE_PYTHON)
    set(Python3_EXECUTABLE "${TORCH_TRACE_PYTHON}")
endif()
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
if(NOT Python3_FOUND)
    _ttc_skip("no Python3 interpreter available")
endif()
if(NOT Python3_Development.Module_FOUND)
    _ttc_skip(
        "no Python3 development headers available for ${Python3_EXECUTABLE}"
    )
endif()
set(_ttc_python "${Python3_EXECUTABLE}")

set(_ttc_torch_config_dir "")
foreach(_ttc_site_packages IN LISTS Python3_SITEARCH Python3_SITELIB)
    if(EXISTS "${_ttc_site_packages}/torch/share/cmake/Torch/TorchConfig.cmake")
        set(_ttc_torch_config_dir "${_ttc_site_packages}/torch/share/cmake/Torch")
        break()
    endif()
endforeach()
if(_ttc_torch_config_dir AND NOT Torch_DIR)
    set(Torch_DIR "${_ttc_torch_config_dir}")
endif()

find_package(Torch CONFIG QUIET)
if(NOT TORCH_FOUND)
    _ttc_skip("find_package(Torch) failed for ${_ttc_python}")
endif()
message(STATUS "torch_trace_collector: find_package(Torch) ${TORCH_INSTALL_PREFIX}")

# Artifact torch_trace_collector-<version>.so; accept 2.11, 2.12, or 2.13.
set(_torch_version "${Torch_VERSION}")
string(REGEX REPLACE "\\+.*$" "" _torch_version "${_torch_version}")
string(REGEX MATCH "^[0-9]+\\.[0-9]+" _torch_minor_version "${_torch_version}")
if(NOT _torch_minor_version MATCHES "^2\\.(11|12|13)$")
    _ttc_skip(
        "Torch ${_torch_version} is not supported (need 2.11, 2.12, or 2.13)"
    )
endif()

set(_torch_includes ${TORCH_INCLUDE_DIRS})
set(_so_name "torch_trace_collector-${_torch_version}")
set(_torch_lib_search_dirs "${TORCH_INSTALL_PREFIX}/lib")

find_library(
    ROCTX_LIB
    NAMES rocprofiler-sdk-roctx
    HINTS
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES lib lib64
)
if(NOT ROCTX_LIB)
    _ttc_skip("rocprofiler-sdk-roctx library not found")
endif()
find_path(
    ROCTX_INCLUDE_DIR
    NAMES rocprofiler-sdk-roctx/roctx.h
    HINTS
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES include
)
if(NOT ROCTX_INCLUDE_DIR)
    _ttc_skip("rocprofiler-sdk-roctx headers not found")
endif()

set(_torch_resolved_libs "")
foreach(_libname IN ITEMS torch torch_cpu c10)
    set(_cachevar "_TORCH_RESOLVED_${_libname}_${_torch_version}")
    find_library(
        ${_cachevar}
        NAMES ${_libname}
        PATHS ${_torch_lib_search_dirs}
        NO_DEFAULT_PATH
    )
    if(NOT ${_cachevar})
        _ttc_skip(
            "lib${_libname}.so not found in any of:\n  ${_torch_lib_search_dirs}"
        )
    endif()
    list(APPEND _torch_resolved_libs ${${_cachevar}})
endforeach()

# torch_python is resolved separately and linked only into the pybind module:
# it leaves Python symbols undefined and names no libpython of its own, so only
# a consumer loaded by an interpreter can resolve them.
set(_torch_python_cachevar "_TORCH_RESOLVED_torch_python_${_torch_version}")
find_library(
    ${_torch_python_cachevar}
    NAMES torch_python
    PATHS ${_torch_lib_search_dirs}
    NO_DEFAULT_PATH
)
if(NOT ${_torch_python_cachevar})
    _ttc_skip(
        "libtorch_python.so not found in any of:\n  ${_torch_lib_search_dirs}"
    )
endif()
set(_torch_python_lib ${${_torch_python_cachevar}})

message(
    STATUS
    "torch_trace_collector: building against ${_ttc_python} "
    "(torch ${_torch_version})"
)

# --- torch-trace-collector library -------------------------------------------------
# PUBLIC so tests inherit Torch/ROCTX includes and libs. torch_python is not linked here.
set(_ttc_headers
    leaf_context.h
    marker_stack.h
    process_state.h
    record_function_callback.h
    record_function_installation.h
    snapshot_store.h
    stack_entry.h
    stats.h
    user_scope.h
    wire_format.h
)
add_library(torch-trace-collector STATIC torch_trace_collector.cpp ${_ttc_headers})

target_include_directories(
    torch-trace-collector
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${_torch_includes} ${ROCTX_INCLUDE_DIR}
)
target_link_libraries(
    torch-trace-collector
    PUBLIC ${_torch_resolved_libs} ${ROCTX_LIB} synchronized gsl_assert
)
target_compile_features(torch-trace-collector PUBLIC cxx_std_17)
target_compile_options(
    torch-trace-collector
    PRIVATE -fvisibility=hidden -Wno-deprecated-declarations
)
set_target_properties(torch-trace-collector PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- pybind module ----------------------------------------------------------
add_library(${_so_name} MODULE torch_trace_collector_module.cpp)
target_link_libraries(
    ${_so_name}
    PRIVATE torch-trace-collector ${_torch_python_lib} Python3::Module
)
target_compile_options(
    ${_so_name}
    PRIVATE -fvisibility=hidden -Wno-deprecated-declarations
)
set_target_properties(${_so_name} PROPERTIES PREFIX "") # torch_trace_collector-<version>.so

# Publish the module target name for the root install rule.
set(TORCH_TRACE_COLLECTOR_TARGET "${_so_name}" CACHE INTERNAL "")

if(ENABLE_TESTS AND TARGET gtest_main)
    add_subdirectory(tests)
endif()

message(STATUS "torch_trace_collector: target ${_so_name} configured")
