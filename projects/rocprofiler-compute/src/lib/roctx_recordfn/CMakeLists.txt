## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
##
## Builds the roctx-recordfn library and the roctx_recordfn-<tag>.so pybind
## module for --torch-trace. Configured in two modes:
##   A. parent add_subdirectory() under ENABLE_TESTS (CI fixture + gtest).
##   B. standalone, driven at runtime by
##      utils/inject_roctx/_backends/torch_cpp_loader.py (JIT build).
set(_rcfn_standalone OFF)
if(NOT PROJECT_NAME)
    cmake_minimum_required(VERSION 3.16)
    project(roctx_recordfn_runtime LANGUAGES CXX)
    set(_rcfn_standalone ON)
endif()

if(NOT _rcfn_standalone AND NOT ENABLE_TESTS)
    return()
endif()

# Interpreter used for the torch/python probe below.
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

if(NOT _rcfn_python)
    message(
        STATUS
        "roctx_recordfn: skipping build (no Python3 interpreter available); "
        "torch-trace tests will skip."
    )
    return()
endif()

# Probe the environment for python/torch metadata and the source fingerprint.
# Exit code 3 means torch is unavailable and the build is skipped.
execute_process(
    COMMAND "${_rcfn_python}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/probe_torch.py"
    OUTPUT_VARIABLE _rcfn_probe
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _rcfn_probe_err
    RESULT_VARIABLE _rcfn_probe_rc
)
if(_rcfn_probe_rc EQUAL 3)
    message(
        STATUS
        "roctx_recordfn: skipping build (torch not importable in "
        "${_rcfn_python}); torch-trace tests will skip."
    )
    return()
elseif(NOT _rcfn_probe_rc EQUAL 0)
    message(
        FATAL_ERROR
        "roctx_recordfn: torch probe failed via ${_rcfn_python}:\n${_rcfn_probe_err}"
    )
endif()

string(REPLACE "\n" ";" _rcfn_probe_lines "${_rcfn_probe}")
list(GET _rcfn_probe_lines 0 _py_major)
list(GET _rcfn_probe_lines 1 _py_minor)
list(GET _rcfn_probe_lines 2 _torch_version)
list(GET _rcfn_probe_lines 3 _torch_includes)
list(GET _rcfn_probe_lines 4 _torch_libs)
list(GET _rcfn_probe_lines 5 _torch_wheel_lib_dir)
list(GET _rcfn_probe_lines 6 _src_fingerprint)

set(_tag "py${_py_major}.${_py_minor}_torch${_torch_version}_src${_src_fingerprint}")
set(_so_name "roctx_recordfn-${_tag}")

message(STATUS "roctx_recordfn: building against ${_rcfn_python} (tag ${_tag})")

# Pin Python3 to the probed interpreter for the module's headers.
set(Python3_EXECUTABLE
    "${_rcfn_python}"
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

# Resolve the torch libraries by absolute path. The wheel lib directory covers
# wheels that omit it from library_paths().
set(_torch_lib_search_dirs ${_torch_libs} ${_torch_wheel_lib_dir})
list(REMOVE_DUPLICATES _torch_lib_search_dirs)

set(_torch_resolved_libs "")
foreach(_libname IN ITEMS torch torch_cpu torch_python c10)
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
    list(APPEND _torch_resolved_libs ${${_cachevar}})
endforeach()

# Detect whether c10::DebugInfoKind accepts a custom string_view-backed key.
# When unavailable, the source falls back to TEST_INFO_2.
include(CheckCXXSourceCompiles)
set(_rcfn_saved_flags "${CMAKE_REQUIRED_FLAGS}")
set(_rcfn_saved_includes "${CMAKE_REQUIRED_INCLUDES}")
set(CMAKE_REQUIRED_FLAGS "-std=c++17")
set(CMAKE_REQUIRED_INCLUDES ${_torch_includes})
check_cxx_source_compiles(
    "#include <c10/util/ThreadLocalDebugInfo.h>
     #include <string_view>
     inline constexpr std::string_view kProbe = \"PROBE\";
     const c10::DebugInfoKind kProbeKind(&kProbe);
     int main() { (void)kProbeKind; return 0; }"
    ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND
)
set(CMAKE_REQUIRED_FLAGS "${_rcfn_saved_flags}")
set(CMAKE_REQUIRED_INCLUDES "${_rcfn_saved_includes}")

# --- roctx-recordfn library -------------------------------------------------
# Usage requirements are PUBLIC so consumers inherit the torch/roctx includes,
# the torch libraries, and the DebugInfoKind flag.
set(_rcfn_headers
    capture_buffer.h
    capture_control.h
    install_state.h
    leaf_context.h
    marker_stack.h
    record_function_bridge.h
    scope_guard.h
    snapshot_store.h
    stack_entry.h
    stats.h
    user_scope.h
    wire_format.h
)
add_library(roctx-recordfn STATIC roctx_recordfn.cpp ${_rcfn_headers})

target_include_directories(
    roctx-recordfn
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${_torch_includes} ${ROCTX_INCLUDE_DIR}
)
target_link_libraries(roctx-recordfn PUBLIC ${_torch_resolved_libs} ${ROCTX_LIB})
target_compile_features(roctx-recordfn PUBLIC cxx_std_17)
target_compile_options(
    roctx-recordfn
    PRIVATE -fvisibility=hidden -Wno-deprecated-declarations
)
set_target_properties(roctx-recordfn PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND)
    target_compile_definitions(
        roctx-recordfn
        PUBLIC ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND=1
    )
    message(STATUS "roctx_recordfn: private DebugInfoKind slot ROCPROF_TORCHTRACE_INFO")
else()
    message(STATUS "roctx_recordfn: legacy enum; using TEST_INFO_2")
endif()

# --- pybind module ----------------------------------------------------------
# The "lib" prefix is omitted; the loader resolves the module by tag.
add_library(${_so_name} MODULE roctx_recordfn_module.cpp)
target_link_libraries(${_so_name} PRIVATE roctx-recordfn Python3::Module)
target_compile_options(
    ${_so_name}
    PRIVATE -fvisibility=hidden -Wno-deprecated-declarations
)
set_target_properties(
    ${_so_name}
    PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
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
