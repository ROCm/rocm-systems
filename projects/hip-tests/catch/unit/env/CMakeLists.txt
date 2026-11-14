# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Environment Variable Tests (AMD only)
if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC
      hipEnvGpuEnablePal.cc
  )

  hip_add_exe_to_target(NAME EnvironmentTest
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests)
endif()
