# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    test_hip_driver_launch_ex_contract.cc
)

hip_add_exe_to_target(NAME ContractDriverLaunchExTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME contract_tests
                      LINKER_LIBS hiprtc::hiprtc)
