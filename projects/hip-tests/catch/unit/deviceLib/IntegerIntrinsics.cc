/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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

TEST_CASE("Unit_IntegerIntrinsics") {
  hipLaunchKernelGGL(compileIntegerIntrinsics, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, 1);
}
