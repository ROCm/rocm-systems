# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    __threadfence_block.cc
    __threadfence.cc
    __threadfence_system.cc
)

hip_add_exe_to_target(NAME ThreadfenceTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
