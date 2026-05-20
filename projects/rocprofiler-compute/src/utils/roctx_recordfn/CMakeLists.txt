## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
##
## Builds roctx_recordfn.so for the --torch-trace C++ tier. Used in two
# modes that produce a byte-name-identical artifact (same tag scheme,
# computed by the loader's _source_fingerprint()):
#   A. as a subdirectory of the parent rocprof-compute build (opt-in
#      via TORCH_TRACE_PREBUILT for packagers with a fixed torch matrix);
#   B. standalone, invoked by inject_roctx_loader.py's cmake tier at
#      runtime in the user's PyTorch env.
if(NOT PROJECT_NAME)
    cmake_minimum_required(VERSION 3.20)
    project(roctx_recordfn_runtime LANGUAGES CXX)
endif()

# Opt-in for packagers baking against a fixed (python, torch, abi)
# cell. OFF by default: rocprof-compute does not pull PyTorch at
# install time, and a prebuilt against the install-time interpreter
# would almost never match the user's runtime env. Invoke as:
#   cmake .. -DTORCH_TRACE_PREBUILT=ON -DTORCH_TRACE_PYTHON=<python>
option(
    TORCH_TRACE_PREBUILT
    "Pre-build roctx_recordfn.so against TORCH_TRACE_PYTHON's torch installation. OFF by default; opt-in for packagers with a fixed torch matrix."
    OFF
)

if(NOT TORCH_TRACE_PREBUILT)
    return()
endif()

# Opt-in path: the operator promised they have a torch-equipped
# Python and want the prebuilt produced. From here, missing torch is
# a real configuration error, not a routine condition -- if you said
# "build the prebuilt" you owe us the inputs to do it.

if(NOT TORCH_TRACE_PYTHON)
    message(
        FATAL_ERROR
        "TORCH_TRACE_PREBUILT=ON requires -DTORCH_TRACE_PYTHON=<python> "
        "pointing at the runtime interpreter you want to bake against. "
        "(The install-time interpreter and the runtime PyTorch env "
        "are almost never the same; pinning is mandatory.)"
    )
endif()

execute_process(
    COMMAND "${TORCH_TRACE_PYTHON}" -c "import torch"
    RESULT_VARIABLE _torch_present_rc
    OUTPUT_QUIET
    ERROR_QUIET
)
if(NOT _torch_present_rc EQUAL 0)
    message(
        FATAL_ERROR
        "TORCH_TRACE_PREBUILT=ON but ${TORCH_TRACE_PYTHON} cannot "
        "import torch. Either install torch into this interpreter, "
        "point TORCH_TRACE_PYTHON at one that has it, or drop the "
        "TORCH_TRACE_PREBUILT=ON flag (the runtime loader will JIT "
        "build at first --torch-trace use)."
    )
endif()

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
# the cache without collision.
set_target_properties(
    ${_so_name}
    PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/prebuilt"
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

target_compile_definitions(${_so_name} PRIVATE _GLIBCXX_USE_CXX11_ABI=${_cxx11_abi})

target_compile_options(
    ${_so_name}
    PRIVATE -fvisibility=hidden -Wno-deprecated-declarations
)

# Install under prebuilt/ next to the source; the runtime loader probes
# this directory first by tag.
install(
    TARGETS ${_so_name}
    LIBRARY
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/utils/roctx_recordfn/prebuilt
        COMPONENT main
)

message(STATUS "roctx_recordfn: pre-built target ${_so_name} configured")
