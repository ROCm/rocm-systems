# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    test_hip_error_api_contract.cc
)

hip_add_exe_to_target(NAME ContractErrorApiTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME contract_tests)
