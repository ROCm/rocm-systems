## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
##
## Builds the torch-trace-collector library and the
## torch_trace_collector-<torch-version>.so pybind module for --torch-trace.
set(BUILD_TORCH_TRACE_COLLECTOR
    "AUTO"
    CACHE STRING
    "Build the torch_trace_collector extension: AUTO, ON (error if torch is absent), OFF"
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

if(BUILD_TORCH_TRACE_COLLECTOR STREQUAL "ON")
    set(_ttc_unavailable_severity FATAL_ERROR)
else()
    set(_ttc_unavailable_severity STATUS)
endif()

# Interpreter used to locate Torch.
if(TORCH_TRACE_PYTHON)
    set(Python3_EXECUTABLE "${TORCH_TRACE_PYTHON}")
endif()
find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_FOUND)
    message(
        ${_ttc_unavailable_severity}
        "torch_trace_collector: no Python3 interpreter available; the extension is "
        "not built."
    )
    return()
endif()
set(_ttc_python "${Python3_EXECUTABLE}")
set(_py_major "${Python3_VERSION_MAJOR}")
set(_py_minor "${Python3_VERSION_MINOR}")

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
    message(
        ${_ttc_unavailable_severity}
        "torch_trace_collector: find_package(Torch) failed for ${_ttc_python}; the "
        "extension is not built."
    )
    return()
endif()
message(STATUS "torch_trace_collector: find_package(Torch) ${TORCH_INSTALL_PREFIX}")

# Artifact name: torch_trace_collector-<torch-version>.so.
# Prefer TORCH_TRACE_TORCH_VERSION from the loader; otherwise use Torch_VERSION.
if(TORCH_TRACE_TORCH_VERSION)
    set(_torch_version "${TORCH_TRACE_TORCH_VERSION}")
else()
    set(_torch_version "${Torch_VERSION}")
endif()
if(NOT _torch_version)
    message(
        ${_ttc_unavailable_severity}
        "torch_trace_collector: Torch version is empty; the extension is not "
        "built."
    )
    return()
endif()

set(_torch_includes ${TORCH_INCLUDE_DIRS})

set(_so_name "torch_trace_collector-${_torch_version}")

message(
    STATUS
    "torch_trace_collector: building against ${_ttc_python} "
    "(torch ${_torch_version})"
)

# Pin Python3 to this interpreter for this directory only. FindPython3
# records its result in private _Python3_* cache entries that outlive this
# directory, so they are saved here and restored after the call below.
set(_ttc_saved_python_vars "")
get_cmake_property(_ttc_cache_vars CACHE_VARIABLES)
foreach(_var IN LISTS _ttc_cache_vars)
    if(_var MATCHES "^_Python3_")
        get_property(_ttc_saved_${_var} CACHE ${_var} PROPERTY VALUE)
        list(APPEND _ttc_saved_python_vars ${_var})
    endif()
endforeach()

set(Python3_EXECUTABLE "${_ttc_python}")
find_package(
    Python3
    ${_py_major}.${_py_minor}
    REQUIRED
    COMPONENTS Interpreter Development.Module
)

foreach(_var IN LISTS _ttc_saved_python_vars)
    set(${_var} "${_ttc_saved_${_var}}" CACHE INTERNAL "")
endforeach()
get_cmake_property(_ttc_cache_vars CACHE_VARIABLES)
foreach(_var IN LISTS _ttc_cache_vars)
    if(_var MATCHES "^_Python3_")
        list(FIND _ttc_saved_python_vars ${_var} _index)
        if(_index EQUAL -1)
            unset(${_var} CACHE)
        endif()
    endif()
endforeach()

get_property(_ttc_python_restored CACHE _Python3_EXECUTABLE PROPERTY VALUE)
if(NOT "${_ttc_python_restored}" STREQUAL "${_ttc_saved__Python3_EXECUTABLE}")
    message(
        FATAL_ERROR
        "torch_trace_collector: _Python3_EXECUTABLE was not restored (expected "
        "'${_ttc_saved__Python3_EXECUTABLE}', got '${_ttc_python_restored}')."
    )
endif()

find_library(
    ROCTX_LIB
    NAMES rocprofiler-sdk-roctx
    HINTS
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES lib lib64
    REQUIRED
)
find_path(
    ROCTX_INCLUDE_DIR
    NAMES rocprofiler-sdk-roctx/roctx.h
    HINTS
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES include
    REQUIRED
)

set(_torch_lib_search_dirs "${TORCH_INSTALL_PREFIX}/lib")

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
        message(
            FATAL_ERROR
            "torch_trace_collector: lib${_libname}.so not found in any of:\n  "
            "${_torch_lib_search_dirs}"
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
    message(
        FATAL_ERROR
        "torch_trace_collector: libtorch_python.so not found in any of:\n  "
        "${_torch_lib_search_dirs}"
    )
endif()
set(_torch_python_lib ${${_torch_python_cachevar}})

# Detect whether c10::DebugInfoKind accepts a custom string_view-backed key.
# When unavailable, the source falls back to TEST_INFO_2. The standard is set
# through CMAKE_CXX_STANDARD rather than CMAKE_REQUIRED_FLAGS: try_compile
# appends its own -std= flag last, which would override anything passed here.
include(CheckCXXSourceCompiles)
set(_ttc_saved_standard "${CMAKE_CXX_STANDARD}")
set(_ttc_saved_standard_required "${CMAKE_CXX_STANDARD_REQUIRED}")
set(_ttc_saved_includes "${CMAKE_REQUIRED_INCLUDES}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_REQUIRED_INCLUDES ${_torch_includes})
check_cxx_source_compiles(
    "#include <c10/util/ThreadLocalDebugInfo.h>
     #include <string_view>
     inline constexpr std::string_view kProbe = \"PROBE\";
     const c10::DebugInfoKind kProbeKind(&kProbe);
     int main() { (void)kProbeKind; return 0; }"
    ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND
)
set(CMAKE_CXX_STANDARD "${_ttc_saved_standard}")
set(CMAKE_CXX_STANDARD_REQUIRED "${_ttc_saved_standard_required}")
set(CMAKE_REQUIRED_INCLUDES "${_ttc_saved_includes}")

# --- torch-trace-collector library -------------------------------------------------
# Usage requirements are PUBLIC so consumers inherit the torch/roctx includes,
# the torch, synchronized and gsl_assert libraries, and the compile definitions.
# torch_python is not among them.
set(_ttc_headers
    leaf_context.h
    marker_stack.h
    process_state.h
    record_function_callback.h
    record_function_installation.h
    scope_guard.h
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

if(ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND)
    target_compile_definitions(
        torch-trace-collector
        PUBLIC ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND=1
    )
    message(
        STATUS
        "torch_trace_collector: private DebugInfoKind slot ROCPROF_TORCHTRACE_INFO"
    )
else()
    message(STATUS "torch_trace_collector: legacy enum; using TEST_INFO_2")
endif()

# --- pybind module ----------------------------------------------------------
# Omit the "lib" prefix. The loader resolves the module by PyTorch version.
add_library(${_so_name} MODULE torch_trace_collector_module.cpp)
target_link_libraries(
    ${_so_name}
    PRIVATE torch-trace-collector ${_torch_python_lib} Python3::Module
)
target_compile_options(
    ${_so_name}
    PRIVATE -fvisibility=hidden -Wno-deprecated-declarations
)
set_target_properties(${_so_name} PROPERTIES PREFIX "")

# Publish the module target name for the root install rule.
set(TORCH_TRACE_COLLECTOR_TARGET "${_so_name}" CACHE INTERNAL "")

if(ENABLE_TESTS AND TARGET gtest_main)
    add_subdirectory(tests)
endif()

message(STATUS "torch_trace_collector: target ${_so_name} configured")
