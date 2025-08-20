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


class HipFunctorTests {
 public:
  // Test that a class functor can be passed to hiplaunchparam
  // and can be used in kernel
  void TestForSimpleClassFunctor(void);
  // Test that a templated class functor can be passed to hiplaunchparam
  // and can be used in kernel
  void TestForClassTemplateFunctor(void);
  // Test that a class functor object ptr  can be passed to hiplaunchparam
  // and can be used in kernel
  void TestForClassObjPtrFunctor(void);
  // Test that a class object containing functor can be passed
  // to hiplaunchparam and can be used in kernel
  void TestForFunctorContainInClassObj(void);
  // Test that a stuct functor can be passed to hiplaunchparam
  // and can be used in kernel
  void TestForSimpleStructFunctor(void);
  // Test that a stuct functor object ptr  can be passed to hiplaunchparam
  // and can be used in kernel
  void TestForStructObjPtrFunctor(void);
  // Test that a templated struct functor can be passed to hiplaunchparam
  // and can be used in kernel
  void TestForStructTemplateFunctor(void);
  // Test that a struct object containing functor can be
  // passed to hiplaunchparam and can be used in kernel
  void TestForFunctorContainInStructObj(void);
};

static const int BLOCK_DIM_SIZE = 1024;
static const int THREADS_PER_BLOCK = 1;

// class functor tests

// Simple doubler Functor
class DoublerFunctor {
 public:
  __device__ int operator()(int x) { return x * 2; }
};

// simple doubler functor passed to kernel
__global__ void DoublerFunctorKernel(DoublerFunctor doubler_, bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int result = doubler_(5);
  deviceResult[x] = (result == 10);
}

void HipFunctorTests::TestForSimpleClassFunctor(void) {
  DoublerFunctor doubler;
  bool *deviceResults, *hostResults;
  HIP_CHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE * sizeof(bool)));
  HIP_CHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE * sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    hostResults[k] = false;
  }

  HIP_CHECK(
      hipMemcpy(deviceResults, hostResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(DoublerFunctorKernel, dim3(BLOCK_DIM_SIZE), dim3(THREADS_PER_BLOCK), 0, 0,
                     doubler, deviceResults);

  // Validation part of TestForSimpleClassFunctor
  HIP_CHECK(
      hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) REQUIRE(hostResults[k] == true);
  HIP_CHECK(hipHostFree(hostResults));
  HIP_CHECK(hipFree(deviceResults));
}

// pointer functor passed to kernel
__global__ void PtrDoublerFunctorKernel(DoublerFunctor* doubler_, bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int result = (*doubler_)(5);
  deviceResult[x] = (result == 10);
}

void HipFunctorTests::TestForClassObjPtrFunctor(void) {
  DoublerFunctor* ptrdoubler = new DoublerFunctor[sizeof(int)];
  bool *deviceResults, *hostResults;
  HIP_CHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE * sizeof(bool)));
  HIP_CHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE * sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    hostResults[k] = false;
  }

  HIP_CHECK(
      hipMemcpy(deviceResults, hostResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(PtrDoublerFunctorKernel, dim3(BLOCK_DIM_SIZE), dim3(THREADS_PER_BLOCK), 0, 0,
                     ptrdoubler, deviceResults);

  // Validation part of TestForClassObjPtrFunctor
  HIP_CHECK(
      hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) REQUIRE(hostResults[k] == true);
  HIP_CHECK(hipHostFree(hostResults));
  HIP_CHECK(hipFree(deviceResults));
  delete[] ptrdoubler;
}

class compare {
 public:
  template <typename T1, typename T2> __device__ bool operator()(const T1& v1, const T2& v2) {
    return v1 > v2;
  }
};

// template functor passed to kernel
__global__ void TemplateFunctorKernel(compare compare_, bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  deviceResult[x] = compare_(2.2, 2.1);
  deviceResult[x] = compare_(2, 1);
  deviceResult[x] = compare_('b', 'a');
}

void HipFunctorTests::TestForClassTemplateFunctor(void) {
  compare comparefunctor;
  bool *deviceResults, *hostResults;
  HIP_CHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE * sizeof(bool)));
  HIP_CHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE * sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    hostResults[k] = false;
  }

  HIP_CHECK(
      hipMemcpy(deviceResults, hostResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(TemplateFunctorKernel, dim3(BLOCK_DIM_SIZE), dim3(THREADS_PER_BLOCK), 0, 0,
                     comparefunctor, deviceResults);

  // Validation part of TestForClassTemplateFunctor
  HIP_CHECK(
      hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) REQUIRE(hostResults[k] == true);
  HIP_CHECK(hipHostFree(hostResults));
  HIP_CHECK(hipFree(deviceResults));
}


// Doubler calculator
class DoublerCalculator {
 public:
  int a, result;
  // fucntor contained in class object
  DoublerFunctor doubler;
};

// doubler functor conatined in class obj passed to kernel
__global__ void DoublerCalculatorFunctorKernel(DoublerCalculator doubler_, bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int result = doubler_.doubler(doubler_.a);
  deviceResult[x] = (doubler_.result == result);
}

void HipFunctorTests::TestForFunctorContainInClassObj(void) {
  DoublerCalculator Doubler;
  bool *deviceResults, *hostResults;
  HIP_CHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE * sizeof(bool)));
  HIP_CHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE * sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    hostResults[k] = false;
  }

  Doubler.a = 5;
  Doubler.result = 10;
  // pass comparefunctor to  hipLaunchParm

  HIP_CHECK(
      hipMemcpy(deviceResults, hostResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(DoublerCalculatorFunctorKernel, dim3(BLOCK_DIM_SIZE), dim3(THREADS_PER_BLOCK),
                     0, 0, Doubler, deviceResults);

  // Validation part of TestForStructTemplateFunctor
  HIP_CHECK(
      hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) REQUIRE(hostResults[k] == true);
  HIP_CHECK(hipHostFree(hostResults));
  HIP_CHECK(hipFree(deviceResults));
}

// Struct functor tests

// Simple doubler Functor
struct sDoublerFunctor {
 public:
  __device__ int operator()(int x) { return x * 2; }
};


// simple sturct doubler functor passed to kernel
__global__ void structDoublerFunctorKernel(sDoublerFunctor doubler_, bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int result = doubler_(5);
  deviceResult[x] = (result == 10);
}

void HipFunctorTests::TestForSimpleStructFunctor(void) {
  sDoublerFunctor doubler;
  bool *deviceResults, *hostResults;
  HIP_CHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE * sizeof(bool)));
  HIP_CHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE * sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    hostResults[k] = false;
  }

  HIP_CHECK(
      hipMemcpy(deviceResults, hostResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(structDoublerFunctorKernel, dim3(BLOCK_DIM_SIZE), dim3(THREADS_PER_BLOCK), 0,
                     0, doubler, deviceResults);

  // Validation part of TestForSimpleStructFunctor
  HIP_CHECK(
      hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) REQUIRE(hostResults[k] == true);
  HIP_CHECK(hipHostFree(hostResults));
  HIP_CHECK(hipFree(deviceResults));
}

// ptr functor passed to kernel
__global__ void structPtrDoublerFunctorKernel(sDoublerFunctor* doubler_, bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int result = (*doubler_)(5);
  deviceResult[x] = (result == 10);
}

void HipFunctorTests::TestForStructObjPtrFunctor(void) {
  sDoublerFunctor* ptrdoubler = new sDoublerFunctor[sizeof(int)];
  bool *deviceResults, *hostResults;
  HIP_CHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE * sizeof(bool)));
  HIP_CHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE * sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    hostResults[k] = false;
  }

  HIP_CHECK(
      hipMemcpy(deviceResults, hostResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(structPtrDoublerFunctorKernel, dim3(BLOCK_DIM_SIZE), dim3(THREADS_PER_BLOCK),
                     0, 0, ptrdoubler, deviceResults);

  // Validation part of TestForStructObjPtrFunctor
  HIP_CHECK(
      hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) REQUIRE(hostResults[k] == true);
  HIP_CHECK(hipHostFree(hostResults));
  HIP_CHECK(hipFree(deviceResults));
  delete[] ptrdoubler;
}

struct sCompare {
 public:
  template <typename T1, typename T2> __device__ bool operator()(const T1& v1, const T2& v2) {
    return v1 > v2;
  }
};

// template functor passed to kernel
__global__ void structTemplateFunctorKernel(sCompare compare_, bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  deviceResult[x] = compare_(2.2, 2.1);
  deviceResult[x] = compare_(2, 1);
  deviceResult[x] = compare_('b', 'a');
}

void HipFunctorTests::TestForStructTemplateFunctor(void) {
  sCompare comparefunctor;
  bool *deviceResults, *hostResults;
  HIP_CHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE * sizeof(bool)));
  HIP_CHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE * sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    hostResults[k] = false;
  }

  HIP_CHECK(
      hipMemcpy(deviceResults, hostResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyHostToDevice));

  // pass comparefunctor to  hipLaunchKernelGGL
  hipLaunchKernelGGL(structTemplateFunctorKernel, dim3(BLOCK_DIM_SIZE), dim3(THREADS_PER_BLOCK), 0,
                     0, comparefunctor, deviceResults);

  // Validation part of TestForStructTemplateFunctor
  HIP_CHECK(
      hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) REQUIRE(hostResults[k] == true);
  HIP_CHECK(hipHostFree(hostResults));
  HIP_CHECK(hipFree(deviceResults));
}

// Doubler calculator struct
struct sDoublerCalculator {
 public:
  int a, result;
  // fucntor contained in class object
  DoublerFunctor doubler;
};


// doubler functor contained in struct passed to kernel
__global__ void DoublerCalculatorFunctorKernel(sDoublerCalculator doubler_, bool* deviceResult) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int result = doubler_.doubler(doubler_.a);
  deviceResult[x] = (doubler_.result == result);
}

void HipFunctorTests::TestForFunctorContainInStructObj(void) {
  sDoublerCalculator Doubler;
  bool *deviceResults, *hostResults;
  HIP_CHECK(hipMalloc(&deviceResults, BLOCK_DIM_SIZE * sizeof(bool)));
  HIP_CHECK(hipHostMalloc(&hostResults, BLOCK_DIM_SIZE * sizeof(bool)));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) {
    // initialize to false, will be set to
    // true if the functor is called in device code
    hostResults[k] = false;
  }

  Doubler.a = 5;
  Doubler.result = 10;
  HIP_CHECK(
      hipMemcpy(deviceResults, hostResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyHostToDevice));


  // pass comparefunctor to  hipLaunchKernelGGL
  hipLaunchKernelGGL(DoublerCalculatorFunctorKernel, dim3(BLOCK_DIM_SIZE), dim3(THREADS_PER_BLOCK),
                     0, 0, Doubler, deviceResults);

  // Validation part of TestForStructTemplateFunctor
  HIP_CHECK(
      hipMemcpy(hostResults, deviceResults, BLOCK_DIM_SIZE * sizeof(bool), hipMemcpyDeviceToHost));
  for (int k = 0; k < BLOCK_DIM_SIZE; ++k) REQUIRE(hostResults[k] == true);
  HIP_CHECK(hipHostFree(hostResults));
  HIP_CHECK(hipFree(deviceResults));
}

/**
* @addtogroup hipLaunchKernelGGL hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

/**
 * Test Description
 * ------------------------
 *    - Test that a class functor can be passed to hiplaunchparam
 * and can be used in kernel.
 *    - Test that a templated class functor can be passed to hiplaunchparam
 * and can be used in kernel.
 *    - Test that a class functor object ptr  can be passed to hiplaunchparam
 * and can be used in kernel.
 *    - Test that a class object containing functor can be passed to hiplaunchparam
 * and can be used in kernel
 *    - Test that a stuct functor can be passed to hiplaunchparam
 * and can be used in kernel
 *    - Test that a stuct functor object ptr  can be passed to hiplaunchparam
 * and can be used in kernel
 *    - Test that a templated struct functor can be passed to hiplaunchparam
 * and can be used in kernel
 *    - Test that a struct object containing functor can be passed to hiplaunchparam
 * and can be used in kernel

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipLaunchParmFunctor.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

TEST_CASE("Unit_hipLaunchParmFunctor") {
  HipFunctorTests FunctorTests;

  SECTION("test for simple class functor") { FunctorTests.TestForSimpleClassFunctor(); }
  SECTION("test for class objptr functor") { FunctorTests.TestForClassObjPtrFunctor(); }
  SECTION("test for class templete functor") { FunctorTests.TestForClassTemplateFunctor(); }
  SECTION("test for simple struct functor") { FunctorTests.TestForSimpleStructFunctor(); }
  SECTION("test for struct objptr functor") { FunctorTests.TestForStructObjPtrFunctor(); }
  SECTION("test for struct templete functor") { FunctorTests.TestForStructTemplateFunctor(); }
  SECTION("test for functor contain in classobj") {
    FunctorTests.TestForFunctorContainInClassObj();
  }
  SECTION("test for functor contain in structobj") {
    FunctorTests.TestForFunctorContainInStructObj();
  }
}

/**
 * End doxygen group KernelTest.
 * @}
 */
