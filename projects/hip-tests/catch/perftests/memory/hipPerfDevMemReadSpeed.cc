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

/**
 * @addtogroup hipMemcpyKernel hipMemcpyKernel
 * @{
 * @ingroup perfMemoryTest
 * `hipMemcpy(void* dst, const void* src, size_t count, hipMemcpyKind kind)` -
 * Copies data between host and device.
 */

// #define ENABLE_DEBUG 1
#include <hip_test_common.hh>

#define ARRAY_SIZE 16

typedef struct d_uint16 {
  uint data[ARRAY_SIZE];
} d_uint16;

__global__ static void read_kernel(d_uint16* src, ulong N, uint* dst) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  uint tmp = 0;
  for (size_t i = idx; i < N; i += stride) {
    for (size_t j = 0; j < ARRAY_SIZE; j++) {
      tmp += src[i].data[j];
    }
  }

  atomicAdd(dst, tmp);
}

static bool hipPerfDevMemReadSpeed_test() {
  d_uint16 *dSrc, *hSrc;
  uint *dDst, *hDst;
  hipStream_t stream;
  ulong N = 4 * 1024 * 1024;
  uint nBytes = N * sizeof(d_uint16);

  int deviceId = 0;
  HIP_CHECK(hipSetDevice(deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

  CONSOLE_PRINT("info: running on bus 0x%x %s with %d CUs\n", props.pciBusID, props.name,
                props.multiProcessorCount);

  const unsigned threadsPerBlock = 64;
  const unsigned blocks = props.multiProcessorCount * 4;

  uint inputData = 0x1;
  int nIter = 1000;

  hSrc = new d_uint16[nBytes];
  REQUIRE(hSrc != nullptr);
  hDst = new uint;
  REQUIRE(hDst != nullptr);
  hDst[0] = 0;

  for (size_t i = 0; i < N; i++) {
    for (int j = 0; j < ARRAY_SIZE; j++) {
      hSrc[i].data[j] = inputData;
    }
  }

  HIP_CHECK(hipMalloc(&dSrc, nBytes));
  HIP_CHECK(hipMalloc(&dDst, sizeof(uint)));

  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemcpy(dSrc, hSrc, nBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dDst, hDst, sizeof(uint), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(read_kernel, dim3(blocks), dim3(threadsPerBlock), 0, stream, dSrc, N, dDst);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(hDst, dDst, sizeof(uint), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  if (hDst[0] != (nBytes / sizeof(uint))) {
    DEBUG_PRINT(
        "hipPerfDevMemReadSpeed - Data validation failed for warm up run! expected %zu got %u\n",
        nBytes / sizeof(uint), hDst[0]);
    return false;
  }

  // measure performance based on host time
  auto all_start = std::chrono::steady_clock::now();

  for (int i = 0; i < nIter; i++) {
    hipLaunchKernelGGL(read_kernel, dim3(blocks), dim3(threadsPerBlock), 0, stream, dSrc, N, dDst);
    HIP_CHECK(hipGetLastError());
  }
  HIP_CHECK(hipDeviceSynchronize());

  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> all_kernel_time = all_end - all_start;

  // read speed in GB/s
  double perf = (static_cast<double>(nBytes * nIter * (1e-09))) / all_kernel_time.count();

  CONSOLE_PRINT(
      "hipPerfDevMemReadSpeed - average read speed of %.2f GB/s achieved for memory size of %u "
      "MB\n",
      perf, nBytes / (1024 * 1024));

  delete[] hSrc;
  delete hDst;
  HIP_CHECK(hipFree(dSrc));
  HIP_CHECK(hipFree(dDst));
  HIP_CHECK(hipStreamDestroy(stream));
  return true;
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipPerfDevMemReadSpeed status.
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfDevMemReadSpeed.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfDevMemReadSpeed_test") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices <= 0) {
    SUCCEED(
        "Skipped testcase hipPerfDevMemReadSpeed as"
        "there is no device to test.");
  } else {
    REQUIRE(true == hipPerfDevMemReadSpeed_test());
  }
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
