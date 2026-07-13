# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    test_hip_capture_to_graph_contract.cc
)

hip_add_exe_to_target(NAME ContractCaptureToGraphTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME contract_tests)
