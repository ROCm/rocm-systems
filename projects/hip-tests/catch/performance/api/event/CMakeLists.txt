# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    hipEventCreate.cc
    hipEventCreateWithFlags.cc
    hipEventRecord.cc
    hipEventDestroy.cc
    hipEventSynchronize.cc
    hipEventElapsedTime.cc
    hipEventQuery.cc
)

hip_add_exe_to_target(NAME EventPerformance
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
