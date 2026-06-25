# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# As CUDA use 32-bit sync masks and not 64-bit this has been disabled on Nvidia
if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC
      warpSync.cc
  )
endif()

hip_add_exe_to_target(NAME WarpSyncPerformance
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
