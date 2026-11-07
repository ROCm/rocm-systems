# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
  trig_funcs.cc
  misc_funcs.cc
  remainder_and_rounding_funcs.cc
  single_precision_intrinsics.cc
  double_precision_intrinsics.cc
  integer_intrinsics.cc
  root_funcs.cc
  log_funcs.cc
  special_funcs.cc
  casting_double_funcs.cc
  casting_float_funcs.cc
  casting_int_funcs.cc
  casting_half2int_funcs.cc
  casting_int2half_funcs.cc
  casting_half_float_funcs.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
  set(LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC}
    pow_funcs.cc
    casting_half2_funcs.cc
    half_precision_math.cc
    half_precision_arithmetic.cc
    half_precision_comparison.cc
  )
  set(LINKER_LIBS hiprtc::hiprtc)
endif()

hip_add_exe_to_target(NAME MathsTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMMON_SHARED_SRC ${COMMON_SHARED_SRC}
                      LINKER_LIBS ${LINKER_LIBS})
