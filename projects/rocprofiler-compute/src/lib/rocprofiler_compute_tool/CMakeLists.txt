## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target rocprofiler-compute-tool)
add_library(
    ${target}
    SHARED
    rocprofiler_compute_tool.h
    rocprofiler_compute_tool.cpp
    counters_writer.h
    counters_writer.cpp
    sdk_callbacks.h
    sdk_callbacks.cpp
    environ_cache.h
    environ_cache.cpp
    input_parameters.h
    input_parameters.cpp
    sdk_wrapper.h
    sdk_wrapper.cpp
    pc_sampling_feature.h
    pc_sampling_feature.cpp
)

target_include_directories(${target} PUBLIC .)
# compression is PRIVATE: not exposed in public headers.
target_link_libraries(
    ${target}
    PUBLIC rocprofiler-sdk::rocprofiler-sdk gsl_assert pc-sampling-collector
    PRIVATE compression
)

# TheRock stamps INSTALL_RPATH on every shared library relative to lib/, but
# this one installs to lib/rocprofiler-compute/, so its lib/rocm_sysdeps/lib
# entry lands one directory short and librocm_sysdeps_z.so.1 goes unresolved.
# Opt out of the stamping and spell out the rpath for the real install dir.
# Drop this once TheRock's post_hook_rocprofiler-compute.cmake sets
# THEROCK_INSTALL_RPATH_ORIGIN for this target, the way the SDK hook already
# does for librocprofiler-sdk-tool.so.
set_target_properties(
    ${target}
    PROPERTIES
        THEROCK_NO_INSTALL_RPATH ON
        INSTALL_RPATH "$ORIGIN:$ORIGIN/..:$ORIGIN/../rocm_sysdeps/lib"
)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
