# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(pmc-amd-smi-provider-tests OBJECT test_provider.cpp)

target_link_libraries(
    pmc-amd-smi-provider-tests
    PUBLIC
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-backend-concepts
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-headers
)

# Always enable both feature flags so NIC and SDMA code paths are always tested.
target_compile_definitions(
    pmc-amd-smi-provider-tests
    PUBLIC ROCPROFSYS_BUILD_AINIC=1 AMD_SMI_SDMA_SUPPORTED=1
)
