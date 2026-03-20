# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    hipKernelLookUpPerf.cc
    hipEventOverFlowPerf.cc
)
hip_add_exe_to_target(NAME perfEventTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME perf_test)
