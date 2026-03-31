# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipMemFaultStackAllocation.cc
    hipLaunchBounds.cc
    hipDynamicShared.cc
    hipDynamicShared2.cc
    hipEmptyKernel.cc
    hipGridLaunch.cc
    hipLanguageExtensions.cc
    hipLaunchParm.cc
    hipLaunchParmFunctor.cc
    hipTestConstant.cc
    hipTestGlobalVariable.cc
    hipTestMemKernel.cc
    hipLaunchBoundsBasic.cc
    hipAutoThreadIdx.cc
    hipLaunchKernelEx.cc
)
if(UNIX)
  set(TEST_SRC ${TEST_SRC}
      hipPrintfKernel.cc)
endif()

# only for AMD
if(HIP_PLATFORM MATCHES "amd")
  set(AMD_SRC
      hipExtLaunchKernelGGLBasic.cc
      hipSetupArgument.cc
      hipConfigureCall.cc
  )
  set(TEST_SRC ${TEST_SRC} ${AMD_SRC})
endif()

hip_add_exe_to_target(NAME KernelTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
