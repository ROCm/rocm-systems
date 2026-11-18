# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    splitBarrier.cc
)

hip_add_exe_to_target(NAME SplitBarrierPerformance
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
