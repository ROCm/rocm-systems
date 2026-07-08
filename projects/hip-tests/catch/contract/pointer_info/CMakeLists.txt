# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC test_hip_pointer_info_contract.cc)

hip_add_exe_to_target(NAME ContractPointerInfoTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME contract_tests)
