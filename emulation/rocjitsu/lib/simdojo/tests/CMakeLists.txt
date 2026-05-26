# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Integration test: verifies find_package(simdojo) for both static and shared.
add_test(
    NAME simdojo_integration_find_package
    COMMAND ${CMAKE_COMMAND}
        -DSIMDOJO_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
        -DTEST_DIR=${CMAKE_CURRENT_SOURCE_DIR}/integration
        -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/_integration
        -P ${CMAKE_CURRENT_SOURCE_DIR}/integration/integration_test.cmake
)
set_tests_properties(simdojo_integration_find_package PROPERTIES
    LABELS "integration"
    TIMEOUT 120
)
