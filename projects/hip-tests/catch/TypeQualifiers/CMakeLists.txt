# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
# <REVIEW> Temporary review marker: this file contains Fabio's changes above the StatCO branch.

# Common Tests
set(TEST_SRC
    hipManagedKeyword.cc
    hipManagedKeywordDeferredInit.cc
)

hip_add_exe_to_target(NAME TypeQualifiers
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
