# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
  set(LINKER_LIBS hiprtc)
endif()

find_package(Boost 1.70.0)
message(STATUS "Boost_FOUND: ${Boost_FOUND}")
if(Boost_FOUND)
  hip_add_exe_to_target(NAME MathsTest
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests COMMON_SHARED_SRC ${COMMON_SHARED_SRC}
                        LINKER_LIBS ${LINKER_LIBS})
  target_include_directories(MathsTest PRIVATE ${Boost_INCLUDE_DIRS})
else()
  message(STATUS "Boost not found. Dependent math tests not enabled.")
endif()
