/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "wavefront_size_gtest.hpp"

#include "../src/util.hpp"

using namespace rocshmem;

__global__ void check_wf_size(int wf_size_prop, int *ret) {
  if (wf_size_prop == WF_SIZE) {
    *ret = 0;
  } else {
    *ret = 1;
  }
}


TEST_F(WavefrontSizeTestFixture, constant_matches_runtime) {
  int device_count = 0;
  hipDeviceProp_t prop;
  int *ret;

  CHECK_HIP(hipGetDeviceCount(&device_count));
  ASSERT_GT(device_count, 0);
  CHECK_HIP(hipHostMalloc(&ret, sizeof(int), 0));

  for (int i = 0; i < device_count; i++) {
    *ret = -1;
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipGetDeviceProperties(&prop, i));

    check_wf_size<<<1, 1>>>(prop.warpSize, ret);
    CHECK_HIP(hipDeviceSynchronize());

    ASSERT_EQ(*ret, 0);
  }
  CHECK_HIP(hipHostFree(ret));
}

