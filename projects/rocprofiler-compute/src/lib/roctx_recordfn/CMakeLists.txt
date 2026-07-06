## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
##
## Builds roctx_recordfn-<tag>.so for --torch-trace.
##   A. parent add_subdirectory() under ENABLE_TESTS (CI fixture).
##   B. standalone (runtime cmake-tier from
##      utils/inject_roctx/_backends/torch_cpp_loader.py).
set(_rcfn_standalone OFF)
if(NOT PROJECT_NAME)
    cmake_minimum_required(VERSION 3.16)
    project(roctx_recordfn_runtime LANGUAGES CXX)
    set(_rcfn_standalone ON)
endif()

if(NOT _rcfn_standalone AND NOT ENABLE_TESTS)
    return()
endif()

# Honor TORCH_TRACE_PYTHON when provided; otherwise locate an interpreter.
if(TORCH_TRACE_PYTHON)
    set(_rcfn_python "${TORCH_TRACE_PYTHON}")
else()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
        set(_rcfn_python "${Python3_EXECUTABLE}")
    else()
        set(_rcfn_python "")
    endif()
endif()

# Skip the build when torch is not importable.
set(_rcfn_torch_ok OFF)
if(_rcfn_python)
    execute_process(
        COMMAND "${_rcfn_python}" -c "import torch"
        RESULT_VARIABLE _rcfn_torch_rc
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(_rcfn_torch_rc EQUAL 0)
        set(_rcfn_torch_ok ON)
    endif()
endif()

if(NOT _rcfn_torch_ok)
    if(_rcfn_python)
        message(
            STATUS
            "roctx_recordfn: skipping .so build (torch not importable in "
            "${_rcfn_python}); torch-trace tests will skip."
        )
    else()
        message(
            STATUS
            "roctx_recordfn: skipping .so build (no Python3 interpreter "
            "available); torch-trace tests will skip."
        )
    endif()
    return()
endif()

# Pin the interpreter for all downstream find_package and probe calls.
set(TORCH_TRACE_PYTHON
    "${_rcfn_python}"
    CACHE FILEPATH
    "Interpreter used to build roctx_recordfn."
    FORCE
)

message(STATUS "roctx_recordfn: building .so against ${TORCH_TRACE_PYTHON}")

execute_process(
    COMMAND
        "${TORCH_TRACE_PYTHON}" -c
        "import sys,torch; \
print(sys.version_info.major); \
print(sys.version_info.minor); \
print(torch.__version__); \
import torch.utils.cpp_extension as e; \
print(';'.join(e.include_paths())); \
print(';'.join(e.library_paths()))"
    OUTPUT_VARIABLE _torch_probe
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_probe_rc
)

if(NOT _torch_probe_rc EQUAL 0)
    message(
        FATAL_ERROR
        "TORCH_TRACE_PYTHON probe failed: ${TORCH_TRACE_PYTHON} could not "
        "report python/torch metadata."
    )
endif()

string(REPLACE "\n" ";" _probe_list "${_torch_probe}")
list(GET _probe_list 0 _py_major)
list(GET _probe_list 1 _py_minor)
list(GET _probe_list 2 _torch_version)
list(GET _probe_list 3 _torch_includes)
list(GET _probe_list 4 _torch_libs)

# Compute the source fingerprint via the loader so build and runtime agree.
execute_process(
    COMMAND
        "${TORCH_TRACE_PYTHON}" -c
        "import sys, pathlib; \
sys.path.insert(0, str(pathlib.Path('${CMAKE_CURRENT_SOURCE_DIR}/../..').resolve())); \
from utils.inject_roctx._backends.torch_cpp_loader import _source_fingerprint; \
print(_source_fingerprint())"
    OUTPUT_VARIABLE _src_fingerprint
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _fp_rc
)
if(NOT _fp_rc EQUAL 0)
    message(
        FATAL_ERROR
        "roctx_recordfn: failed to compute source fingerprint via "
        "${TORCH_TRACE_PYTHON}."
    )
endif()

set(_tag "py${_py_major}.${_py_minor}_torch${_torch_version}_src${_src_fingerprint}")
set(_so_name "roctx_recordfn-${_tag}")

# Pin Python3 lookup to the same interpreter as TORCH_TRACE_PYTHON.
set(Python3_EXECUTABLE
    "${TORCH_TRACE_PYTHON}"
    CACHE FILEPATH
    "Interpreter used to build roctx_recordfn."
    FORCE
)
find_package(
    Python3
    ${_py_major}.${_py_minor}
    REQUIRED
    COMPONENTS Interpreter Development.Module
)

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

add_library(${_so_name} MODULE roctx_recordfn.cpp)

# The "lib" prefix is omitted; the loader resolves the module by tag.
set_target_properties(
    ${_so_name}
    PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
    ${_so_name}
    PRIVATE ${_torch_includes} ${ROCTX_INCLUDE_DIR} ${Python3_INCLUDE_DIRS}
)

# Resolve torch libraries by absolute path. The wheel lib directory is
# added as a fallback because some wheels omit it from library_paths().
message(STATUS "roctx_recordfn: torch lib dirs reported by probe: ${_torch_libs}")

execute_process(
    COMMAND
        "${TORCH_TRACE_PYTHON}" -c
        "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE _torch_wheel_lib_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(_torch_lib_search_dirs ${_torch_libs} ${_torch_wheel_lib_dir})
list(REMOVE_DUPLICATES _torch_lib_search_dirs)
message(
    STATUS
    "roctx_recordfn: torch lib search dirs (probe + wheel fallback): ${_torch_lib_search_dirs}"
)

set(_torch_required_libs torch torch_cpu torch_python c10)
set(_torch_resolved_libs "")
foreach(_libname IN LISTS _torch_required_libs)
    set(_cachevar "_TORCH_RESOLVED_${_libname}_${_tag}")
    find_library(
        ${_cachevar}
        NAMES ${_libname}
        PATHS ${_torch_lib_search_dirs}
        NO_DEFAULT_PATH
    )
    if(NOT ${_cachevar})
        message(
            FATAL_ERROR
            "roctx_recordfn: lib${_libname}.so not found in any of:\n  "
            "${_torch_lib_search_dirs}"
        )
    endif()
    message(STATUS "roctx_recordfn: resolved ${_libname} -> ${${_cachevar}}")
    list(APPEND _torch_resolved_libs ${${_cachevar}})
endforeach()

target_link_libraries(${_so_name} PRIVATE ${_torch_resolved_libs} ${ROCTX_LIB})

# Detect whether c10::DebugInfoKind accepts a custom string_view-backed
# key. When unavailable, the source falls back to TEST_INFO_2.
include(CheckCXXSourceCompiles)
set(_rcfn_saved_required_flags ${CMAKE_REQUIRED_FLAGS})
set(_rcfn_saved_required_libs ${CMAKE_REQUIRED_LIBRARIES})
set(_rcfn_saved_required_includes ${CMAKE_REQUIRED_INCLUDES})
get_target_property(_rcfn_inc_dirs ${_so_name} INCLUDE_DIRECTORIES)
set(CMAKE_REQUIRED_INCLUDES ${_rcfn_inc_dirs})
set(CMAKE_REQUIRED_FLAGS "-std=c++17")
check_cxx_source_compiles(
    "#include <c10/util/ThreadLocalDebugInfo.h>
     #include <string_view>
     inline constexpr std::string_view kProbe = \"PROBE\";
     const c10::DebugInfoKind kProbeKind(&kProbe);
     int main() { (void)kProbeKind; return 0; }"
    ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND
)
set(CMAKE_REQUIRED_FLAGS ${_rcfn_saved_required_flags})
set(CMAKE_REQUIRED_LIBRARIES ${_rcfn_saved_required_libs})
set(CMAKE_REQUIRED_INCLUDES ${_rcfn_saved_required_includes})
if(ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND)
    target_compile_definitions(
        ${_so_name}
        PRIVATE ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND=1
    )
    message(STATUS "roctx_recordfn: private DebugInfoKind slot ROCPROF_TORCHTRACE_INFO")
else()
    message(STATUS "roctx_recordfn: legacy enum; using TEST_INFO_2")
endif()

target_compile_options(
    ${_so_name}
    PRIVATE -fvisibility=hidden -Wno-deprecated-declarations
)

if(NOT _rcfn_standalone)
    install(
        TARGETS ${_so_name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${CMAKE_PROJECT_NAME} COMPONENT main
    )
endif()

if(NOT _rcfn_standalone AND ENABLE_TESTS AND TARGET gtest_main)
    add_subdirectory(tests)
endif()

message(STATUS "roctx_recordfn: target ${_so_name} configured")
