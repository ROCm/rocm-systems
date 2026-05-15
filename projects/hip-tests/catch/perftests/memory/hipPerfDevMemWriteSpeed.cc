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

#include <hip_test_common.hh>

#define ARRAY_SIZE 16

typedef struct d_uint16 {
  uint data[ARRAY_SIZE];
} d_uint16;

__global__ void write_kernel(d_uint16* dst, ulong N, d_uint16 pval) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < N; i += stride) {
    dst[i] = pval;
  }
}

static bool hipPerfDevMemWriteSpeed_test() {
  d_uint16 *dDst, *hDst;
  ulong N = 4 * 1024 * 1024;
  uint nBytes = N * sizeof(d_uint16);

  uint inputData = 0xabababab;
  int nIter = 1000;
  d_uint16 pval;

  int deviceId = 0;
  HIP_CHECK(hipSetDevice(deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

  CONSOLE_PRINT("info: running on bus 0x%x %s with %d CUs\n", props.pciBusID, props.name,
                props.multiProcessorCount);

  const unsigned threadsPerBlock = 64;
  const unsigned blocks = props.multiProcessorCount * 4;

  for (int i = 0; i < ARRAY_SIZE; i++) {
    pval.data[i] = inputData;
  }

  hDst = new d_uint16[nBytes];
  REQUIRE(hDst != nullptr);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < ARRAY_SIZE; j++) {
      hDst[i].data[j] = 0;
    }
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMalloc(&dDst, nBytes));
  hipLaunchKernelGGL(write_kernel, dim3(blocks), dim3(threadsPerBlock), 0, stream, dDst, N, pval);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(hDst, dDst, nBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < ARRAY_SIZE; j++) {
      if (hDst[i].data[j] != inputData) {
        DEBUG_PRINT(
            "hipPerfDevMemWriteSpeed - Data validation failed for warm up run! at index i: %u "
            "element j: %u expected 0x%x but got 0x%x\n",
            i, j, inputData, hDst[i].data[j]);
        return false;
      }
    }
  }

  // measure performance based on host time
  auto all_start = std::chrono::steady_clock::now();

  for (int i = 0; i < nIter; i++) {
    hipLaunchKernelGGL(write_kernel, dim3(blocks), dim3(threadsPerBlock), 0, stream, dDst, N, pval);
    HIP_CHECK(hipGetLastError());
  }
  HIP_CHECK(hipDeviceSynchronize());

  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> all_kernel_time = all_end - all_start;

  // read speed in GB/s
  double perf = (static_cast<double>(nBytes * nIter * (1e-09))) / all_kernel_time.count();

  CONSOLE_PRINT(
      "hipPerfDevMemWriteSpeed - average write speed of %.2f GB/s achieved for memory size of %u "
      "MB\n",
      perf, nBytes / (1024 * 1024));

  delete[] hDst;
  HIP_CHECK(hipFree(dDst));
  HIP_CHECK(hipStreamDestroy(stream));
  return true;
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipPerfDevMemWriteSpeed status.
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfDevMemWriteSpeed.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfDevMemWriteSpeed_test") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices <= 0) {
    SUCCEED(
        "Skipped testcase hipPerfDevMemWriteSpeed as"
        "there is no device to test.");
  } else {
    REQUIRE(true == hipPerfDevMemWriteSpeed_test());
  }
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
