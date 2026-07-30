# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    kernel_launch_common.cc
    triple_chevron.cc
    hipLaunchKernel.cc
    hipLaunchCooperativeKernel.cc
)

if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC ${TEST_SRC}
        hipExtLaunchKernel.cc
        hipExtLaunchKernelGGL.cc
    )
endif()

hip_add_exe_to_target(NAME KernelLaunchPerformance
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
