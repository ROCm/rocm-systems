# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    hip_memory_pool_contract.cc
)

hip_add_exe_to_target(NAME ContractMemoryPoolTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME contract_tests)
