# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(pmc-amd-smi-provider-tests OBJECT test_provider.cpp)

target_link_libraries(
    pmc-amd-smi-provider-tests
    PUBLIC
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-amd-smi-backend
        rocprofiler-systems::rocprofiler-systems-compile-definitions
)

target_include_directories(
    pmc-amd-smi-provider-tests
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        # Reuse mock_backend.hpp from the backend layer tests
        ${PROJECT_SOURCE_DIR}/source/lib/backends/amd_smi/tests
)

# Always enable both feature flags so NIC and SDMA code paths are always tested.
target_compile_definitions(
    pmc-amd-smi-provider-tests
    PUBLIC ROCPROFSYS_BUILD_AINIC=1 AMD_SMI_SDMA_SUPPORTED=1
)
