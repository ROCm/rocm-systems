# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# CP Dynamic Data Prefetch Performance Tests
set(TEST_SRC
    hipPerfDynDataPrefetch.cc
)

hip_add_exe_to_target(NAME PrefetchPerformance
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
