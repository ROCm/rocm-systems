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

#include <string.h>
#include <math.h>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>

#include <algorithm>
#include <type_traits>

using namespace std;
////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
////////////////////////////////////////////////////////////////////////////////

bool verifyBitwise(...) { return true; }

template <typename T, typename enable_if<is_integral<T>{}>::type* = nullptr>
bool verifyBitwise(T* gpuData, int len) {
  // Atomic and
  T val = 0xff;
  for (int i = 0; i < len; ++i) {
    // 9th element should be 1
    val &= (2 * i + 7);
  }
  REQUIRE(val == gpuData[8]);

  // atomic Or
  val = 0;
  for (int i = 0; i < len; ++i) {
    // 10th element should be 0xff
    val |= (1 << i);
  }
  REQUIRE(val == gpuData[9]);

  // atomic Xor
  val = 0xff;

  for (int i = 0; i < len; ++i) {
    // 11th element should be 0xff
    val ^= i;
  }

  REQUIRE(val == gpuData[10]);
  return true;
}

bool verifySub(...) { return true; }

template <typename T,
          typename enable_if<is_same<T, int>{} || is_same<T, unsigned int>{}>::type* = nullptr>
bool verifySub(T* gpuData, int len) {
  T val = 0;

  for (int i = 0; i < len; ++i) {
    val -= 10;
  }

  REQUIRE(val == gpuData[1]);
  return true;
}

bool verifyExch(...) { return true; }

template <typename T, typename enable_if<!is_same<T, double>{}>::type* = nullptr>  // NOLINT
bool computeExchExch(T* gpuData, int len) {
  T val = 0;

  for (T i = 0; i < len; ++i) {
    if (i == gpuData[2]) {
      return true;
      break;
    }
  }
}

bool VerifyIntegral(...) { return true; }

template <typename T, typename enable_if<is_integral<T>{}>::type* = nullptr>
bool VerifyIntegral(T* gpuData, int len) {
  // atomic Max
  T val = 0;
  for (int i = 0; i < len; ++i) {
    // fourth element should be len-1
    val = max(val, static_cast<T>(i));
  }

  REQUIRE(val == gpuData[3]);

  // atomic Min
  val = 1 << 8;

  for (int i = 0; i < len; ++i) {
    val = min(val, static_cast<T>(i));
  }

  REQUIRE(val == gpuData[4]);

  // atomic Inc
  T limit = 17;
  val = 0;

  for (int i = 0; i < len; ++i) {
    val = (val >= limit) ? 0 : val + 1;
  }

  REQUIRE(val == gpuData[5]);

  // atomic Dec
  limit = 137;
  val = 0;

  for (int i = 0; i < len; ++i) {
    val = ((val == 0) || (val > limit)) ? limit : val - 1;
  }

  REQUIRE(val == gpuData[6]);

  // atomic CAS
  for (int i = 0; i < len; ++i) {
    // eighth element should be a member of [0, len)
    if (static_cast<T>(i) == gpuData[7]) {
      return true;
      break;
    }
  }
  return verifyBitwise(gpuData, len) && verifySub(gpuData, len);
}

template <typename T> bool verifyData(T* gpuData, int len) {
  T val = 0;
  for (int i = 0; i < len; ++i) {
    val += 10;
  }

  REQUIRE(val == gpuData[0]);
  return VerifyIntegral(gpuData, len) && verifyExch(gpuData, len);
}

__device__ void testKernelExch(...) {}

template <typename T, typename enable_if<!is_same<T, double>{}>::type* = nullptr>
__device__ void testKernelExch(T* g_odata) {
  // access thread id
  const T tid = blockDim.x * blockIdx.x + threadIdx.x;

  // Atomic exchange
  atomicExch(&g_odata[2], tid);
}

__device__ void testKernelSub(...) {}

template <typename T,
          typename enable_if<is_same<T, int>{} || is_same<T, unsigned int>{}>::type* = nullptr>
__device__ void testKernelSub(T* g_odata) {
  // Atomic subtraction (final should be 0)
  atomicSub(&g_odata[1], 10);
}

__device__ void testKernelIntegral(...) {}

template <typename T, typename enable_if<is_integral<T>{}>::type* = nullptr>
__device__ void testKernelIntegral(T* g_odata) {
  // access thread id
  const T tid = blockDim.x * blockIdx.x + threadIdx.x;

  // Atomic maximum
  atomicMax(&g_odata[3], tid);

  // Atomic minimum
  atomicMin(&g_odata[4], tid);

  // Atomic increment (modulo 17+1)
  atomicInc((unsigned int*)&g_odata[5], 17);

  // Atomic decrement
  atomicDec((unsigned int*)&g_odata[6], 137);

  // Atomic compare-and-swap
  atomicCAS(&g_odata[7], tid - 1, tid);

  // Bitwise atomic instructions

  // Atomic AND
  atomicAnd(&g_odata[8], 2 * tid + 7);

  // Atomic OR
  atomicOr(&g_odata[9], 1 << tid);

  // Atomic XOR
  atomicXor(&g_odata[10], tid);

  testKernelSub(g_odata);
}

template <typename T> __global__ void testKernel(T* g_odata) {
  // Atomic addition
  atomicAdd(&g_odata[0], 10);
  testKernelIntegral(g_odata);
  testKernelExch(g_odata);
}

template <typename T> static void runTest() {
  bool testResult = true;
  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  unsigned int numData = 11;
  unsigned int memSize = sizeof(T) * numData;

  // allocate mem for the result on host side
  T* hOData = reinterpret_cast<T*>(malloc(memSize));

  // initialize the memory
  for (unsigned int i = 0; i < numData; i++) {
    hOData[i] = 0;
  }
  // To make the AND and XOR tests generate something other than 0...
  hOData[8] = hOData[10] = 0xff;

  // allocate device memory for result
  T* dOData;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dOData), memSize));
  // copy host memory to device to initialize to zero
  HIP_CHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));

  // execute the kernel
  hipLaunchKernelGGL(testKernel, dim3(numBlocks), dim3(numThreads), 0, 0, dOData);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult == verifyData(hOData, numThreads * numBlocks));

  // Cleanup memory
  free(hOData);
  HIP_CHECK(hipFree(dOData));
}

TEST_CASE("Unit_SimpleAtomicsTest") {
  SECTION("test for int") { runTest<int>(); }
  SECTION("test for unsigned int") { runTest<unsigned int>(); }
  SECTION("test for float") { runTest<float>(); }
#if HT_AMD
  SECTION("test for unsigned long long") { runTest<uint64_t>(); }
  SECTION("test for double") { runTest<double>(); }
#endif
}
