# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    hipExtCopyEngine.cc
)
hip_add_exe_to_target(NAME CopyEngineTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      PROPERTY CXX_STANDARD 17)
