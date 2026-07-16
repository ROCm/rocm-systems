# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    hipMemset.cc
    hipMemsetAsync.cc
    hipMemsetD8.cc
    hipMemsetD8Async.cc
    hipMemsetD16.cc
    hipMemsetD16Async.cc
    hipMemsetD32.cc
    hipMemsetD32Async.cc
    hipMemset2D.cc
    hipMemset2DAsync.cc
    hipMemset3D.cc
    hipMemset3DAsync.cc
)

hip_add_exe_to_target(NAME MemsetPerformance
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
