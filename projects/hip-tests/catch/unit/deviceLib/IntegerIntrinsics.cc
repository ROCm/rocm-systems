/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip/device_functions.h>
#include <algorithm>

#pragma GCC diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wunused-variable"

__device__ void integer_intrinsics() {
  __brev((unsigned int)10);
  __brevll((uint64_t)10);
  __byte_perm((unsigned int)0, (unsigned int)0, 0);
  __clz(static_cast<int>(10));
  __clzll((int64_t)10);
  __ffs(static_cast<int>(10));
  __ffsll((long long)(10));  // NOLINT
  __funnelshift_l((unsigned int)0xfacefeed, (unsigned int)0xdeadbeef, 0);
  __funnelshift_lc((unsigned int)0xfacefeed, (unsigned int)0xdeadbeef, 0);
  __funnelshift_r((unsigned int)0xfacefeed, (unsigned int)0xdeadbeef, 0);
  __funnelshift_rc((unsigned int)0xfacefeed, (unsigned int)0xdeadbeef, 0);
  __hadd(static_cast<int>(1), static_cast<int>(3));
  __mul24(static_cast<int>(1), static_cast<int>(2));
  __mul64hi((int64_t)1, (int64_t)2);
  __mulhi(static_cast<int>(1), static_cast<int>(2));
  __popc((unsigned int)4);
  __popcll((uint64_t)4);
  int a = min(static_cast<int>(4), static_cast<int>(5));
  int b = max(static_cast<int>(4), static_cast<int>(5));
  __rhadd(static_cast<int>(1), static_cast<int>(2));
  __sad(static_cast<int>(1), static_cast<int>(2), 0);
  __uhadd((unsigned int)1, (unsigned int)3);
  __umul24((unsigned int)1, (unsigned int)2);
  __umul64hi((uint64_t)1, (uint64_t)2);
  __umulhi((unsigned int)1, (unsigned int)2);
  __urhadd((unsigned int)1, (unsigned int)2);
  __usad((unsigned int)1, (unsigned int)2, 0);

  assert(1);
}

__global__ void compileIntegerIntrinsics(int) { integer_intrinsics(); }

HIP_TEST_CASE(Unit_IntegerIntrinsics) {
  hipLaunchKernelGGL(compileIntegerIntrinsics, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, 1);
}
