## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
##
## Builds roctx_recordfn.so for the --torch-trace C++ tier. Two consumers,
# both producing a byte-name-identical artifact (same tag scheme,
# computed by the loader's _source_fingerprint()):
#   A. parent rocprof-compute project's add_subdirectory() -- gated on
#      ENABLE_TESTS. The .so is a CI test fixture so pytest can pin the
#      loader to a known artifact via ROCPROFCOMPUTE_ROCTX_RECORDFN_SO
#      and never shell out to cmake / torch.utils.cpp_extension at test
#      time. Production installs ship sources only (parent's
#      install(DIRECTORY src/lib/roctx_recordfn ...) under
#      ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/lib/roctx_recordfn/)
#      and the runtime loader's cmake-tier builds the .so on first
#      --torch-trace use in the user's PyTorch env.
#   B. standalone, invoked by inject_roctx_loader.py's cmake-tier at
#      runtime. Always proceeds; this is the production degraded path.
set(_rcfn_standalone OFF)
if(NOT PROJECT_NAME)
    cmake_minimum_required(VERSION 3.20)
    project(roctx_recordfn_runtime LANGUAGES CXX)
    set(_rcfn_standalone ON)
endif()

# Parent-project mode: gate on ENABLE_TESTS. Tests are the only in-tree
# consumer; production installs rely on the runtime loader. Standalone
# mode always proceeds because the loader is its only caller.
if(NOT _rcfn_standalone AND NOT ENABLE_TESTS)
    return()
endif()

# Resolve the interpreter we'll build the .so against. Explicit
# -DTORCH_TRACE_PYTHON wins (set by the loader's cmake-tier); otherwise
# pick whatever find_package(Python3) returns.
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

# Probe torch in the resolved interpreter. Missing torch is a soft skip
# in both modes: torch-trace tests skip via the loader's None return,
# and the standalone runtime caller already handles None gracefully.
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

# Pin TORCH_TRACE_PYTHON to the resolved interpreter so every downstream
# find_package / probe consults the same Python.
set(TORCH_TRACE_PYTHON
    "${_rcfn_python}"
    CACHE FILEPATH
    "Pinned to the interpreter whose torch we built against."
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
print(int(torch._C._GLIBCXX_USE_CXX11_ABI)); \
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
        "report (python, torch, abi). This is a real configuration error "
        "(torch imported but the introspection script crashed); please "
        "report with the CMake log."
    )
endif()

string(REPLACE "\n" ";" _probe_list "${_torch_probe}")
list(GET _probe_list 0 _py_major)
list(GET _probe_list 1 _py_minor)
list(GET _probe_list 2 _torch_version)
list(GET _probe_list 3 _cxx11_abi)
list(GET _probe_list 4 _torch_includes)
list(GET _probe_list 5 _torch_libs)

# Compute the source-content fingerprint via the loader's own
# _source_fingerprint(). Keeping the formula in exactly one place
# means a future change to what counts as a "build input" (e.g.
# adding a private header) automatically rotates both the runtime
# resolved filename AND this install-time produced filename in lock
# step -- no risk of the two drifting and silently breaking the
# prebuilt tier.
execute_process(
    COMMAND
        "${TORCH_TRACE_PYTHON}" -c
        "import sys, pathlib; \
sys.path.insert(0, str(pathlib.Path('${CMAKE_CURRENT_SOURCE_DIR}/../..').resolve())); \
from utils.inject_roctx_loader import _source_fingerprint; \
print(_source_fingerprint())"
    OUTPUT_VARIABLE _src_fingerprint
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _fp_rc
)
if(NOT _fp_rc EQUAL 0)
    message(
        FATAL_ERROR
        "roctx_recordfn: failed to compute source fingerprint via "
        "${TORCH_TRACE_PYTHON}; cannot guarantee the prebuilt .so "
        "filename matches what the runtime loader resolves."
    )
endif()

set(_tag
    "py${_py_major}.${_py_minor}_torch${_torch_version}_abi${_cxx11_abi}_src${_src_fingerprint}"
)
set(_so_name "roctx_recordfn-${_tag}")

# Pin find_package(Python3 ...) to TORCH_TRACE_PYTHON so the pybind11
# Development.Module headers come from the same interpreter we probed
# for torch/abi above.
set(Python3_EXECUTABLE
    "${TORCH_TRACE_PYTHON}"
    CACHE FILEPATH
    "Pinned to TORCH_TRACE_PYTHON for the roctx_recordfn prebuilt"
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

# Emit "roctx_recordfn-${_tag}.so" (no "lib" prefix, tag in the
# filename). The loader resolves by tag via spec_from_file_location,
# so a bare "roctx_recordfn.so" would be invisible to it; keeping the
# tag also lets multiple (python, torch, abi, src) cells coexist in
# the cache without collision. Output flat under the build dir so the
# loader's cmake-tier (which builds in a fresh tmp dir and reads the
# .so back from there) doesn't need to know about a sub-folder.
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

# Resolve each torch .so by absolute path via find_library() instead
# of `-L<dir> -ltorch`. ROCm wheels have been observed to report
# library_paths() directories that lack the expected symlinks, which
# makes the linker fail opaquely; doing the resolution in CMake lets
# us search the wheel's own lib dir as a fallback and fail configure
# with a clear message naming the missing .so.
message(STATUS "roctx_recordfn: torch lib dirs reported by probe: ${_torch_libs}")

# Always fold in the wheel's own lib dir as a fallback. library_paths()
# is supposed to include it but some stripped ROCm nightlies have been
# observed to either omit it or report a sibling directory that lacks
# the libtorch*.so set; we'd rather over-search than fail opaquely.
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
    # Per-library cache var so a later configure pass can't reuse a
    # stale find from a different torch wheel. The tag (which includes
    # the torch version + ABI + source fingerprint) makes the cache key
    # specific to this exact wheel.
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
            "roctx_recordfn: could not find lib${_libname}.so in any of "
            "the torch lib search dirs:\n  ${_torch_lib_search_dirs}\n"
            "This is unusual -- libtorch wheels normally ship "
            "libtorch.so, libtorch_cpu.so, libtorch_python.so, and "
            "libc10.so under torch/lib/. If this is a stripped or "
            "partial wheel, the runtime loader cannot build the .so "
            "and torch_trace will fall back to the Python-only "
            "injector. Verify with: \n"
            "  ${TORCH_TRACE_PYTHON} -c \"import os, torch; "
            "p=os.path.join(os.path.dirname(torch.__file__), 'lib'); "
            "print(p, sorted(os.listdir(p)))\""
        )
    endif()
    message(STATUS "roctx_recordfn: resolved ${_libname} -> ${${_cachevar}}")
    list(APPEND _torch_resolved_libs ${${_cachevar}})
endforeach()

target_link_libraries(${_so_name} PRIVATE ${_torch_resolved_libs} ${ROCTX_LIB})

# Feature probe: PyTorch >= 2.7-ish ships c10::DebugInfoKind as a class
# constructible from `const std::string_view*`, which lets downstream
# tools allocate their own private TLDI slot by pointer identity. Older
# PyTorch ships it as an `enum class` with fixed members; in that case
# we fall back to TEST_INFO_2 (used only inside PyTorch's own gtest
# suite, dormant in production training/inference workloads).
include(CheckCXXSourceCompiles)
set(_rcfn_saved_required_flags ${CMAKE_REQUIRED_FLAGS})
set(_rcfn_saved_required_libs ${CMAKE_REQUIRED_LIBRARIES})
set(_rcfn_saved_required_includes ${CMAKE_REQUIRED_INCLUDES})
get_target_property(_rcfn_inc_dirs ${_so_name} INCLUDE_DIRECTORIES)
set(CMAKE_REQUIRED_INCLUDES ${_rcfn_inc_dirs})
set(CMAKE_REQUIRED_FLAGS "-std=c++17 -D_GLIBCXX_USE_CXX11_ABI=${_cxx11_abi}")
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
    message(
        STATUS
        "roctx_recordfn: using private DebugInfoKind slot "
        "ROCPROF_TORCHTRACE_INFO (pointer-identity)"
    )
else()
    message(
        STATUS
        "roctx_recordfn: PyTorch DebugInfoKind is the legacy enum form; "
        "falling back to c10::DebugInfoKind::TEST_INFO_2 (safe in "
        "training/inference workloads, may collide with PyTorch's own "
        "gtest suite if profiled directly)"
    )
endif()

target_compile_definitions(${_so_name} PRIVATE _GLIBCXX_USE_CXX11_ABI=${_cxx11_abi})

target_compile_options(
    ${_so_name}
    PRIVATE -fvisibility=hidden -Wno-deprecated-declarations
)

# Parent-project mode: expose the artifact path to the parent so it can
# pin torch-trace CTest entries via ROCPROFCOMPUTE_ROCTX_RECORDFN_SO.
# Standalone mode (the loader's runtime cmake-tier) reads the .so back
# directly from the build dir and does not need this export.
#
# No install(TARGETS ...) rule: the .so is a CI test fixture, not a
# production artifact. Production installs ship the sources (parent
# CMakeLists install(DIRECTORY src/lib/roctx_recordfn ...)) and the
# runtime loader builds the .so on first --torch-trace use.
if(NOT _rcfn_standalone)
    set(ROCTX_RECORDFN_SO_PATH "$<TARGET_FILE:${_so_name}>" PARENT_SCOPE)
endif()

message(STATUS "roctx_recordfn: target ${_so_name} configured")
