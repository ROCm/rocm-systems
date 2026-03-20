# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipPerfDotProduct.cc
    hipPerfMandelbrot.cc
)

hip_add_exe_to_target(NAME perfComputeTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME perf_test)
