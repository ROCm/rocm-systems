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
      __hip_atomic_fetch_min.cc
      __hip_atomic_fetch_max.cc
      atomic_builtins.cc
      acquire_release.cc
      sequential_consistency.cc
      atomicAdd.cc
      atomicAdd_system.cc
      safeAtomicAdd.cc
      atomicSub.cc
      atomicSub_system.cc
      atomicCAS.cc
      atomicCAS_system.cc
      __hip_atomic_fetch_add.cc
      __hip_atomic_compare_exchange_strong.cc
      atomicExch.cc
      atomicExch_system.cc
      __hip_atomic_fetch_and.cc
      __hip_atomic_fetch_or.cc
      __hip_atomic_fetch_xor.cc
      __hip_atomic_exchange.cc
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

  if(UNIX)
    set(EXPECTED_ERRORS 40)

    file(GLOB NEGATIVE_TEST_SRC
        "atomicAnd_negative_kernels.cc"
        "atomicOr_negative_kernels.cc"
        "atomicXor_negative_kernels.cc"
        "atomicMin_negative_kernels.cc"
        "atomicMax_negative_kernels.cc"
        "atomic_builtin_kernels.cc"
        "atomicAdd_negative_kernels.cc"
        "atomicSub_negative_kernels.cc"
        "atomicInc_negative_kernels.cc"
        "atomicDec_negative_kernels.cc"
        "atomicCAS_negative_kernels.cc"
        "atomicExch_negative_kernels.cc"
        "atomicExch_system_negative_kernels.cc")

    file(COPY ${NEGATIVE_TEST_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_SRC_FILES ${NEGATIVE_TEST_SRC})
  endif()
endif()
