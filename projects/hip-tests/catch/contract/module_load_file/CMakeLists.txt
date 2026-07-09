# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC test_hip_module_load_file_contract.cc)

hip_add_exe_to_target(NAME ContractModuleLoadFileTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME contract_tests)
