/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>

#include "roctx.h"

#include <thread>

#define HIP_CALL(call)                                                                             \
  do {                                                                                             \
    hipError_t err = call;                                                                         \
    if (err != hipSuccess) {                                                                       \
      fprintf(stderr, "%s\n", hipGetErrorString(err));                                             \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

__global__ void kernel() {}

int main(int argc, char* argv[]) {
  HIP_CALL(hipSetDevice(0));

  // Not in a roctx range.
  kernel<<<1, 1>>>();

  int ret = roctxRangePush("NestedRangeA");

  // In a simple first level roctx range.
  kernel<<<1, 1>>>();

  if (roctxRangePop() != ret) return -1;

  roctxRangePush("NestedRangeB");
  roctxRangePush("NestedRangeC");
  roctx_range_id_t id = roctxRangeStart("StartStopRangeA");

  // In a nested roctx range.
  kernel<<<1, 1>>>();

  roctxRangePop();
  roctxRangePop();

  std::thread thread([id]() { roctxRangeStop(id); });
  thread.join();

  roctxRangePush("NestedRangeD");
  roctxRangePush("NestedRangeE");
  roctxRangePop();

  // In a first level roctx range, but after a nested range.
  kernel<<<1, 1>>>();

  if (roctxRangePop() != 0) return -1;

  HIP_CALL(hipDeviceSynchronize());
  return 0;
}
