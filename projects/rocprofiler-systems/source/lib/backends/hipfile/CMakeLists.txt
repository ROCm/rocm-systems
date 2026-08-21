# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_library(rocprofiler-systems-hipfile-backend INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-hipfile-backend
    ALIAS rocprofiler-systems-hipfile-backend
)

# Link libhipfile directly via its imported target (hip::hipfile), which carries
# both the public headers and the shared library. This matches how the profiler
# consumes amd_smi and other ROCm libraries.
target_link_libraries(
    rocprofiler-systems-hipfile-backend
    INTERFACE
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-rocm
        hip::hipfile
)

target_compile_definitions(
    rocprofiler-systems-hipfile-backend
    INTERFACE ROCPROFSYS_BUILD_HIPFILE=1
)
