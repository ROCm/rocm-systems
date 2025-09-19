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
// #define ENABLE_DEBUG 1
#define sharedMemSize1 2048
#define sharedMemSize2 256

__global__ void sharedMemReadSpeed1(float* outBuf, ulong N) {
  size_t gid = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t lid = threadIdx.x;
  __shared__ float local[sharedMemSize1];

  float val1 = 0;
  float val2 = 0;
  float val3 = 0;
  float val4 = 0;

  for (int i = 0; i < (sharedMemSize1 / 64); i++) {
    local[lid + i * 64] = lid;
  }

  __syncthreads();

  val1 += local[lid];
  val2 += local[lid + 64];
  val3 += local[lid + 128];
  val4 += local[lid + 192];
  val1 += local[lid + 256];
  val2 += local[lid + 320];
  val3 += local[lid + 384];
  val4 += local[lid + 448];
  val1 += local[lid + 512];
  val2 += local[lid + 576];
  val3 += local[lid + 640];
  val4 += local[lid + 704];
  val1 += local[lid + 768];
  val2 += local[lid + 832];
  val3 += local[lid + 896];
  val4 += local[lid + 960];
  val1 += local[lid + 1024];
  val2 += local[lid + 1088];
  val3 += local[lid + 1152];
  val4 += local[lid + 1216];
  val1 += local[lid + 1280];
  val2 += local[lid + 1344];
  val3 += local[lid + 1408];
  val4 += local[lid + 1472];
  val1 += local[lid + 1536];
  val2 += local[lid + 1600];
  val3 += local[lid + 1664];
  val4 += local[lid + 1728];
  val1 += local[lid + 1792];
  val2 += local[lid + 1856];
  val3 += local[lid + 1920];
  val4 += local[lid + 1984];

  if (gid < N) {
    outBuf[gid] = val1 + val2 + val3 + val4;
  }
}

__global__ void sharedMemReadSpeed2(float* outBuf, ulong N) {
  size_t gid = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t lid = threadIdx.x;
  __shared__ float local[sharedMemSize2];

  float val0 = 0.0f;
  float val1 = 0.0f;

  for (int i = 0; i < (sharedMemSize2 / 64); i++) {
    local[lid + i * 64] = lid;
  }

  __syncthreads();

#pragma nounroll
  for (uint i = 0; i < 32; i++) {
    val0 += local[8 * i + 0];
    val1 += local[8 * i + 1];
    val0 += local[8 * i + 2];
    val1 += local[8 * i + 3];
    val0 += local[8 * i + 4];
    val1 += local[8 * i + 5];
    val0 += local[8 * i + 6];
    val1 += local[8 * i + 7];
  }

  if (gid < N) {
    outBuf[gid] = val0 + val1;
  }
}

static bool hipPerfSharedMemReadSpeed_test() {
  float* dDst;
  float* hDst;
  hipStream_t stream;
  constexpr uint numSizes = 4;
  constexpr uint Sizes[numSizes] = {262144, 1048576, 4194304, 16777216};
  uint numReads1 = 32;
  uint numReads2 = 256;
  uint sharedMemSizeBytes1 = sharedMemSize1 * sizeof(float);
  uint sharedMemSizeBytes2 = sharedMemSize2 * sizeof(float);
  int nIter = 1000;
  const unsigned threadsPerBlock = 64;

  static int device = 0;
  HIP_CHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  CONSOLE_PRINT("info: running on bus 0x%x %s with %d CUs\n", props.pciBusID, props.name,
                props.multiProcessorCount);

  HIP_CHECK(hipStreamCreate(&stream));

  for (int nTest = 0; nTest < numSizes; nTest++) {
    uint nBytes = Sizes[nTest % numSizes];
    ulong N = nBytes / sizeof(float);
    const unsigned blocks = N / threadsPerBlock;

    hDst = new float[nBytes];
    HIP_CHECK(hDst == 0 ? hipErrorOutOfMemory : hipSuccess);
    memset(hDst, 0, nBytes);

    HIP_CHECK(hipMalloc(&dDst, nBytes));
    HIP_CHECK(hipMemcpy(dDst, hDst, nBytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(sharedMemReadSpeed1, dim3(blocks), dim3(threadsPerBlock), 0, stream, dDst,
                       N);
    HIP_CHECK(hipMemcpy(hDst, dDst, nBytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    int tmp = 0;
    for (int i = 0; i < N; i++) {
      if (i % threadsPerBlock == 0) {
        tmp = 0;
      }
      if (hDst[i] != tmp) {
        DEBUG_PRINT("Data validation failed for warm up run! expected %d got %f\n", tmp, hDst[i]);
        return false;
      }
      tmp += threadsPerBlock / 2;
    }

    auto all_start = std::chrono::steady_clock::now();
    for (int i = 0; i < nIter; i++) {
      hipLaunchKernelGGL(sharedMemReadSpeed1, dim3(blocks), dim3(threadsPerBlock), 0, stream, dDst,
                         N);
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto all_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> all_kernel_time = all_end - all_start;

    // read speed in GB/s
    double perf = (static_cast<double>(blocks * threadsPerBlock) *
                   (numReads1 * sizeof(float) + sharedMemSizeBytes1 / 64) * nIter * (1e-09)) /
                  all_kernel_time.count();

    CONSOLE_PRINT(
        "info: read speed = %.2f GB/s for %d KB shared memory with %d threads, %d reads in "
        "sharedMemReadSpeed1 kernel\n",
        perf, sharedMemSizeBytes1 / 1024, blocks * threadsPerBlock, numReads1);

    delete[] hDst;
    HIP_CHECK(hipFree(dDst));
  }

  for (int nTest = 0; nTest < numSizes; nTest++) {
    uint nBytes = Sizes[nTest % numSizes];
    ulong N = nBytes / sizeof(float);
    const unsigned blocks = N / threadsPerBlock;

    hDst = new float[nBytes];
    HIP_CHECK(hDst == 0 ? hipErrorOutOfMemory : hipSuccess);
    memset(hDst, 0, nBytes);

    HIP_CHECK(hipMalloc(&dDst, nBytes));
    HIP_CHECK(hipMemcpy(dDst, hDst, nBytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(sharedMemReadSpeed2, dim3(blocks), dim3(threadsPerBlock), 0, stream, dDst,
                       N);
    HIP_CHECK(hipMemcpy(hDst, dDst, nBytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    auto all_start = std::chrono::steady_clock::now();
    for (int i = 0; i < nIter; i++) {
      hipLaunchKernelGGL(sharedMemReadSpeed2, dim3(blocks), dim3(threadsPerBlock), 0, stream, dDst,
                         N);
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto all_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> all_kernel_time = all_end - all_start;

    // read speed in GB/s
    double perf = (static_cast<double>(blocks * threadsPerBlock) *
                   (numReads2 * sizeof(float) + sharedMemSizeBytes2 / 64) * nIter * (1e-09)) /
                  all_kernel_time.count();

    CONSOLE_PRINT(
        "info: read speed = %.2f GB/s for %d KB shared memory with %d threads, %d reads in "
        "sharedMemReadSpeed2 kernel\n",
        perf, sharedMemSizeBytes2 / 1024, blocks * threadsPerBlock, numReads2);

    delete[] hDst;
    HIP_CHECK(hipFree(dDst));
  }
  HIP_CHECK(hipStreamDestroy(stream));
  return true;
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipPerfSharedMemReadSpeed status.
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfSharedMemReadSpeed.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfSharedMemReadSpeed_test") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices <= 0) {
    SUCCEED(
        "Skipped testcase hipPerfSharedMemReadSpeed as"
        "there is no device to test.\n");
  } else {
    REQUIRE(true == hipPerfSharedMemReadSpeed_test());
  }
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
