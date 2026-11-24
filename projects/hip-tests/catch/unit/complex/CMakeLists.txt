# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Common Tests - Test independent of all platforms

set(TEST_SRC
    complex.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
  set(LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(LINKER_LIBS hiprtc::hiprtc)
endif()

hip_add_exe_to_target(NAME ComplexTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${LINKER_LIBS})
