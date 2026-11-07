# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC
      atomicAnd.cc
      atomicAnd_system.cc
      atomicOr.cc
      atomicOr_system.cc
      atomicXor.cc
      atomicXor_system.cc
      atomicMin.cc
      atomicMin_system.cc
      atomicMax.cc
      atomicMax_system.cc
      unsafeAtomicMin.cc
      unsafeAtomicMax.cc
      atomicAdd.cc
      atomicAdd_system.cc
      safeAtomicAdd.cc
      atomicSub.cc
      atomicSub_system.cc
      atomicCAS.cc
      atomicCAS_system.cc
      atomicExch.cc
      atomicExch_system.cc
      atomicInc.cc
      atomicDec.cc
    )

 if(NOT ENABLE_SPIRV)
    set(SPIRV_INCOMPATIBLE_TESTS
      safeAtomicMin.cc
      safeAtomicMax.cc
      unsafeAtomicAdd.cc
    )
    set(TEST_SRC ${TEST_SRC} ${SPIRV_INCOMPATIBLE_TESTS})
  endif()

  hip_add_exe_to_target(NAME AtomicsTest
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests
                        LINKER_LIBS hiprtc::hiprtc)
endif()
