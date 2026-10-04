# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
  gl2_scaling.cc
)

hip_add_exe_to_target(NAME ResidencyPerformance
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
