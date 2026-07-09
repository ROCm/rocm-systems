# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC test_hip_driver_entry_point_contract.cc)

hip_add_exe_to_target(NAME ContractDriverEntryPointTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME contract_tests)
