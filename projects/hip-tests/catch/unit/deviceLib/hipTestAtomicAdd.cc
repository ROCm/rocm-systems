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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios :
 (TestCase 1)::
 1) Execute atomicAdd in multi threaded scenario by diverging the data across
 multiple threads and validate the output at the end of all operations.
 2) Execute atomicAddNoRet in multi threaded scenario by diverging the data
 across multiple threads and validate the output at the end of all operations.
 (TestCase 2)::
 3) Execute atomicAdd API and validate the result.
 4) Execute atomicAddNoRet API and validate the result.
 (TestCase 3)::
 5) atomicadd/NoRet negative scenarios (TBD).
*/

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
/*
 * Defines initial and increment values
 */
#define INCREMENT_VALUE 10
#define INT_INITIAL_VALUE 10
#define FLOAT_INITIAL_VALUE 10.50
#define DOUBLE_INITIAL_VALUE 200.12
#define LONG_INITIAL_VALUE 10000
#define UNSIGNED_INITIAL_VALUE 20

#if HT_NVIDIA
// atomicAddNoRet is unavailable in cuda
template <typename T> __device__ void atomicAddNoRet(T* x, int y) {
  atomicAdd(x, static_cast<T>(y));
}
#endif

bool p_atomicNoRet = false;

template <typename T> __global__ void atomicnoret_manywaves(T* C_d) {
  atomicAddNoRet(C_d, INCREMENT_VALUE);
}

template <typename T> __global__ void atomic_manywaves(T* C_d) { atomicAdd(C_d, INCREMENT_VALUE); }

template <typename T> __global__ void atomicnoret_simple(T* C_d) {
  atomicAddNoRet(C_d, INCREMENT_VALUE);
}

template <typename T> __global__ void atomic_simple(T* C_d) { atomicAdd(C_d, INCREMENT_VALUE); }

template <typename T> bool atomictest_manywaves(const T& initial_val) {
  unsigned int ThreadsperBlock = 10;
  unsigned int numBlocks = 1;
  T memSize = sizeof(T);
  T* hOData = reinterpret_cast<T*>(malloc(memSize));
  *hOData = initial_val;
  T* dOData;
  HIP_CHECK(hipMalloc(&dOData, memSize));
  // copy host memory to device to initialize to zero
  HIP_CHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));

  // execute the kernel
  hipLaunchKernelGGL(atomic_manywaves, dim3(numBlocks), dim3(ThreadsperBlock), 0, 0, dOData);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));
  REQUIRE(hOData[0] ==
          initial_val + static_cast<T>(INCREMENT_VALUE * (ThreadsperBlock * numBlocks)));

  // Cleanup memory
  free(hOData);
  HIP_CHECK(hipFree(dOData));

  return true;
}

template <typename T> bool atomictestnoret_manywaves(const T& initial_val) {
  unsigned int ThreadsperBlock = 10;
  unsigned int numBlocks = 1;
  T memSize = sizeof(T);
  T* hOData = reinterpret_cast<T*>(malloc(memSize));
  *hOData = initial_val;
  T* dOData;
  HIP_CHECK(hipMalloc(&dOData, memSize));
  // copy host memory to device to initialize to zero
  HIP_CHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));

  // execute the kernel
  hipLaunchKernelGGL(atomicnoret_manywaves, dim3(numBlocks), dim3(ThreadsperBlock), 0, 0, dOData);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));
  REQUIRE(hOData[0] == initial_val + (INCREMENT_VALUE * (ThreadsperBlock * numBlocks)));

  // Cleanup memory
  free(hOData);
  HIP_CHECK(hipFree(dOData));

  return true;
}

template <typename T> bool atomictest_simple(const T& initial_val) {
  unsigned int ThreadsperBlock = 1;
  unsigned int numBlocks = 1;
  T memSize = sizeof(T);
  T* hOData = reinterpret_cast<T*>(malloc(memSize));
  *hOData = initial_val;
  T* dOData;
  HIP_CHECK(hipMalloc(&dOData, memSize));
  // copy host memory to device to initialize to zero
  HIP_CHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));

  // execute the kernel
  hipLaunchKernelGGL(atomic_simple, dim3(numBlocks), dim3(ThreadsperBlock), 0, 0, dOData);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));
  REQUIRE(hOData[0] == initial_val + INCREMENT_VALUE);

  // Cleanup memory
  free(hOData);
  HIP_CHECK(hipFree(dOData));

  return true;
}

template <typename T> bool atomictestnoret_simple(const T& initial_val) {
  unsigned int ThreadsperBlock = 1;
  unsigned int numBlocks = 1;
  T memSize = sizeof(T);
  T* hOData = reinterpret_cast<T*>(malloc(memSize));
  *hOData = initial_val;
  T* dOData;
  HIP_CHECK(hipMalloc(&dOData, memSize));
  // copy host memory to device to initialize to zero
  HIP_CHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));

  // execute the kernel
  hipLaunchKernelGGL(atomicnoret_simple, dim3(numBlocks), dim3(ThreadsperBlock), 0, 0, dOData);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));
  REQUIRE(hOData[0] == initial_val + INCREMENT_VALUE);

  // Cleanup memory
  free(hOData);
  HIP_CHECK(hipFree(dOData));

  return true;
}

TEST_CASE("Unit_hipTestAtomicAdd") {
  bool TestPassed = true;

  SECTION("atomic tests with many waves") {
    REQUIRE(TestPassed == atomictest_manywaves<int>(INT_INITIAL_VALUE));
    REQUIRE(TestPassed == atomictest_manywaves<unsigned int>(UNSIGNED_INITIAL_VALUE));
    REQUIRE(TestPassed == atomictest_manywaves<float>(FLOAT_INITIAL_VALUE));
#if HT_AMD
    REQUIRE(TestPassed == atomictest_manywaves<uint64_t>(LONG_INITIAL_VALUE));
    REQUIRE(TestPassed == atomictest_manywaves<double>(DOUBLE_INITIAL_VALUE));
#endif
  }
  SECTION("atomic tests with many waves and no return") {
    REQUIRE(TestPassed == atomictestnoret_manywaves<float>(FLOAT_INITIAL_VALUE));
  }
  SECTION("simple atomic tests") {
    REQUIRE(TestPassed == atomictest_simple<int>(INT_INITIAL_VALUE));
    REQUIRE(TestPassed == atomictest_simple<unsigned int>(UNSIGNED_INITIAL_VALUE));
    REQUIRE(TestPassed == atomictest_simple<float>(FLOAT_INITIAL_VALUE));
#if HT_AMD
    REQUIRE(TestPassed == atomictest_simple<uint64_t>(LONG_INITIAL_VALUE));
    REQUIRE(TestPassed == atomictest_simple<double>(DOUBLE_INITIAL_VALUE));
#endif
  }
  SECTION("Simple atomic test with no return") {
    REQUIRE(TestPassed == atomictestnoret_simple<float>(FLOAT_INITIAL_VALUE));
  }
}
