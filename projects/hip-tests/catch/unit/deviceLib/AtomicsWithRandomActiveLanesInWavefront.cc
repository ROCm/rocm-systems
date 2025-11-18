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

// Testcase Description:
// Verifies working of following Global Atomic Operations:
// 1. atomicAdd,
// 2. atomicSub,
// 3. atomicMax,
// 4. atomicMin,
// 5. atomicAnd,
// 6. atomicOr, and
// 7. atomicXor
// for INT, UNSIGNED and FLOAT input types.

// Tests consist of following scenarios:
// 1) Uniform input value to atomic operation (i.e. every thread will be
//    performing atomic operation with the same value)
// 2) Divergent input value (i.e. every thread with be performing
//    atomic operation on different values)

#include <hip_test_common.hh>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
////////////////////////////////////////////////////////////////////////////////

// Atomic Operations
enum AtomicOp : unsigned {
  Add,
  Sub,
  Max,
  Min,
  And,
  Or,
  Xor,
};

// Initial values for atomic ops and various types
template <typename T, AtomicOp Op> struct Initial {
  constexpr static T value;
};

template <> const int Initial<int, AtomicOp::Add>::value = 0;
template <> const unsigned Initial<unsigned, AtomicOp::Add>::value = 0;
template <> const float Initial<float, AtomicOp::Add>::value = 0.0f;

template <> const int Initial<int, AtomicOp::Sub>::value = 0;
template <> const unsigned Initial<unsigned, AtomicOp::Sub>::value = 0;
template <> const float Initial<float, AtomicOp::Sub>::value = 0.0f;

template <> const int Initial<int, AtomicOp::Min>::value = INT_MAX;
template <> const unsigned Initial<unsigned, AtomicOp::Min>::value = UINT_MAX;
template <> const float Initial<float, AtomicOp::Min>::value = std::numeric_limits<float>::min();

template <> const int Initial<int, AtomicOp::Max>::value = INT_MIN;
template <> const unsigned Initial<unsigned, AtomicOp::Max>::value = 0;
template <> const float Initial<float, AtomicOp::Max>::value = std::numeric_limits<float>::max();

template <> const int Initial<int, AtomicOp::And>::value = 0xffffffff;
template <> const unsigned Initial<unsigned, AtomicOp::And>::value = 0xffffffff;

template <> const int Initial<int, AtomicOp::Or>::value = 0;
template <> const unsigned Initial<unsigned, AtomicOp::Or>::value = 0;

template <> const int Initial<int, AtomicOp::Xor>::value = 0x5a5a5a5a;
template <> const unsigned Initial<unsigned, AtomicOp::Xor>::value = 0x5a5a5a5a;


// Uniform values for atomic ops and various types
template <typename T, AtomicOp Op> struct Uniform {
  constexpr static T value;
};

template <> const int Uniform<int, AtomicOp::Add>::value = 10;
template <> const unsigned Uniform<unsigned, AtomicOp::Add>::value = 10;
template <> const float Uniform<float, AtomicOp::Add>::value = 10.0f;

template <> const int Uniform<int, AtomicOp::Sub>::value = 10;
template <> const unsigned Uniform<unsigned, AtomicOp::Sub>::value = 10;
template <> const float Uniform<float, AtomicOp::Sub>::value = 10.0f;

template <> const int Uniform<int, AtomicOp::Min>::value = 10;
template <> const unsigned Uniform<unsigned, AtomicOp::Min>::value = 10;
template <> const float Uniform<float, AtomicOp::Min>::value = 10.0f;

template <> const int Uniform<int, AtomicOp::Max>::value = 10;
template <> const unsigned Uniform<unsigned, AtomicOp::Max>::value = 10;
template <> const float Uniform<float, AtomicOp::Max>::value = 10.0f;

template <> const int Uniform<int, AtomicOp::And>::value = 10;
template <> const unsigned Uniform<unsigned, AtomicOp::And>::value = 10;

template <> const int Uniform<int, AtomicOp::Or>::value = 10;
template <> const unsigned Uniform<unsigned, AtomicOp::Or>::value = 10;

template <> const int Uniform<int, AtomicOp::Xor>::value = 10;
template <> const unsigned Uniform<unsigned, AtomicOp::Xor>::value = 10;

// Auto-verification APIs for uniform values for various atomic ops
template <typename T> bool verifyAdd(T* gpuData, int len, bool* activeLanes) {
  T val = Initial<T, AtomicOp::Add>::value;
  T uniformValue = Uniform<T, AtomicOp::Add>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val += uniformValue;
  }
  return val == gpuData[0];
}

template <typename T> bool verifySub(T* gpuData, int len, bool* activeLanes) {
  T val = Initial<T, AtomicOp::Sub>::value;
  T uniformValue = Uniform<T, AtomicOp::Sub>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val -= uniformValue;
  }
  return val == gpuData[1];
}

template <typename T> bool verifyMax(T* gpuData, int len, bool* activeLanes) {
  T val = Initial<T, AtomicOp::Max>::value;
  T uniformValue = Uniform<T, AtomicOp::Max>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val = std::max<T>(val, uniformValue);
  }
  return val == gpuData[2];
}

template <typename T> bool verifyMin(T* gpuData, int len, bool* activeLanes) {
  T val = Initial<T, AtomicOp::Min>::value;
  T uniformValue = Uniform<T, AtomicOp::Min>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val = std::min<T>(val, uniformValue);
  }
  return val == gpuData[3];
}

template <typename T> bool verifyAnd(T* gpuData, int len, bool* activeLanes) {
  T val = Initial<T, AtomicOp::And>::value;
  T uniformValue = Uniform<T, AtomicOp::And>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val &= uniformValue;
  }
  return val == gpuData[4];
}

template <typename T> bool verifyOr(T* gpuData, int len, bool* activeLanes) {
  T val = Initial<T, AtomicOp::Or>::value;
  T uniformValue = Uniform<T, AtomicOp::Or>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val |= uniformValue;
  }
  return val == gpuData[5];
}

template <typename T> bool verifyXor(T* gpuData, int len, bool* activeLanes) {
  T val = Initial<T, AtomicOp::Xor>::value;
  T uniformValue = Uniform<T, AtomicOp::Xor>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val ^= uniformValue;
  }
  return val == gpuData[6];
}

// Auto-verification APIs for divergent values for various atomic ops
template <typename T>
bool verifyAdd_divValue(T* gpuData, int len, bool* activeLanes, T* divergentValue) {
  T val = Initial<T, AtomicOp::Add>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val += divergentValue[i];
  }
  if (std::is_same<T, float>::value) {
    REQUIRE(val == Approx(gpuData[0]));
    return true;
  }
  return val == gpuData[0];
}

template <typename T>
bool verifySub_divValue(T* gpuData, int len, bool* activeLanes, T* divergentValue) {
  T val = Initial<T, AtomicOp::Sub>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val -= divergentValue[i];
  }
  if (std::is_same<T, float>::value) {
    REQUIRE(val == Approx(gpuData[1]));
    return true;
  }
  return val == gpuData[1];
}

template <typename T>
bool verifyMax_divValue(T* gpuData, int len, bool* activeLanes, T* divergentValue) {
  T val = Initial<T, AtomicOp::Max>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val = std::max<T>(val, divergentValue[i]);
  }

  if (std::is_same<T, float>::value) {
    REQUIRE(val == Approx(gpuData[2]));
    return true;
  }
  return val == gpuData[2];
}

template <typename T>
bool verifyMin_divValue(T* gpuData, int len, bool* activeLanes, T* divergentValue) {
  T val = Initial<T, AtomicOp::Min>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val = std::min<T>(val, divergentValue[i]);
  }

  if (std::is_same<T, float>::value) {
    REQUIRE(val == Approx(gpuData[3]));
    return true;
  }
  return val == gpuData[3];
}

template <typename T>
bool verifyAnd_divValue(T* gpuData, int len, bool* activeLanes, T* divergentValue) {
  T val = Initial<T, AtomicOp::And>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val &= ~(1 << divergentValue[i]);
  }
  return val == gpuData[4];
}

template <typename T>
bool verifyOr_divValue(T* gpuData, int len, bool* activeLanes, T* divergentValue) {
  T val = Initial<T, AtomicOp::Or>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val |= (1 << divergentValue[i]);
  }
  return val == gpuData[5];
}

template <typename T>
bool verifyXor_divValue(T* gpuData, int len, bool* activeLanes, T* divergentValue) {
  T val = Initial<T, AtomicOp::Xor>::value;
  for (int i = 0; i < len; ++i) {
    if (activeLanes[i]) val ^= divergentValue[i];
  }
  return val == gpuData[6];
}

// Kernels exercising atomic ops on uniform value for random set of active lanes in wavefront
template <typename T> __global__ void testAtomicAdd_uniValue(T* g_odata, bool* g_activelanes) {
  T uniformValue = Uniform<T, AtomicOp::Add>::value;
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicAdd(&g_odata[0], uniformValue);
}

template <typename T> __global__ void testAtomicSub_uniValue(T* g_odata, bool* g_activelanes) {
  T uniformValue = Uniform<T, AtomicOp::Sub>::value;
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicSub(&g_odata[1], uniformValue);
}


template <typename T> __global__ void testAtomicMax_uniValue(T* g_odata, bool* g_activelanes) {
  T uniformValue = Uniform<T, AtomicOp::Max>::value;
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicMax(&g_odata[2], uniformValue);
}

template <typename T> __global__ void testAtomicMin_uniValue(T* g_odata, bool* g_activelanes) {
  T uniformValue = Uniform<T, AtomicOp::Min>::value;
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicMin(&g_odata[3], uniformValue);
}


template <typename T> __global__ void testAtomicAnd_uniValue(T* g_odata, bool* g_activelanes) {
  T uniformValue = Uniform<T, AtomicOp::And>::value;
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicAnd(&g_odata[4], uniformValue);
}

template <typename T> __global__ void testAtomicOr_uniValue(T* g_odata, bool* g_activelanes) {
  T uniformValue = Uniform<T, AtomicOp::Or>::value;
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicOr(&g_odata[5], uniformValue);
}

template <typename T> __global__ void testAtomicXor_uniValue(T* g_odata, bool* g_activelanes) {
  T uniformValue = Uniform<T, AtomicOp::Xor>::value;
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicXor(&g_odata[6], uniformValue);
}

// Kernels exercising atomic ops on divergent values for random set of active lanes in wavefront
template <typename T>
__global__ void testAtomicAdd_divValue(T* g_odata, bool* g_activelanes, T* g_divergentvalue) {
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicAdd(&g_odata[0], g_divergentvalue[tid]);
  ;
}

template <typename T>
__global__ void testAtomicSub_divValue(T* g_odata, bool* g_activelanes, T* g_divergentvalue) {
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicSub(&g_odata[1], g_divergentvalue[tid]);
}


template <typename T>
__global__ void testAtomicMax_divValue(T* g_odata, bool* g_activelanes, T* g_divergentvalue) {
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicMax(&g_odata[2], g_divergentvalue[tid]);
}

template <typename T>
__global__ void testAtomicMin_divValue(T* g_odata, bool* g_activelanes, T* g_divergentvalue) {
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicMin(&g_odata[3], g_divergentvalue[tid]);
}


template <typename T>
__global__ void testAtomicAnd_divValue(T* g_odata, bool* g_activelanes, T* g_divergentvalue) {
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicAnd(&g_odata[4], ~(1 << g_divergentvalue[tid]));
}

template <typename T>
__global__ void testAtomicOr_divValue(T* g_odata, bool* g_activelanes, T* g_divergentvalue) {
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicOr(&g_odata[5], (1 << g_divergentvalue[tid]));
}

template <typename T>
__global__ void testAtomicXor_divValue(T* g_odata, bool* g_activelanes, T* g_divergentvalue) {
  const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (g_activelanes[tid]) atomicXor(&g_odata[6], g_divergentvalue[tid]);
}


template <typename T> static void runIntTest() {
  bool testResult = true;
  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  // This test exercises 7 atomic operations
  // 1. atomicAdd
  // 2. atomicSub
  // 3. atomicMax
  // 4. atomicMin
  // 5. atomicAnd
  // 6. atomicOr
  // 7. atomicXor
  unsigned int numData = 7;
  unsigned int memSize = sizeof(T) * numData;
  unsigned int N = numThreads * numBlocks;

  // allocate mem for the result on host side
  T* hOData = reinterpret_cast<T*>(malloc(memSize));
  bool* hIActiveLanes = (bool*)(malloc(N * sizeof(bool)));

  // initialize the memory
  hOData[0] = Initial<T, AtomicOp::Add>::value;
  hOData[1] = Initial<T, AtomicOp::Sub>::value;
  hOData[2] = Initial<T, AtomicOp::Max>::value;
  hOData[3] = Initial<T, AtomicOp::Min>::value;
  hOData[4] = Initial<T, AtomicOp::And>::value;
  hOData[5] = Initial<T, AtomicOp::Or>::value;
  hOData[6] = Initial<T, AtomicOp::Xor>::value;


  for (unsigned int i = 0; i < N; i++) {
    // Genearte random values betwenn 0 to 9 to get random active and inactive lanes for atomic
    // operations
    hIActiveLanes[i] = std::rand() / ((RAND_MAX + 1u) / 9) % 2 == 0;
  }

  // allocate device memory for result
  T* dOData;
  bool* dIActiveLanes;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dOData), memSize));
  HIP_CHECK(hipMalloc(reinterpret_cast<bool**>(&dIActiveLanes), N * sizeof(bool)));

  // copy host memory to device to initialize to zero
  HIP_CHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dIActiveLanes, hIActiveLanes, N, hipMemcpyHostToDevice));

  // execute the atomicAdd kernel
  hipLaunchKernelGGL(testAtomicAdd_uniValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult == verifyAdd(hOData, numThreads * numBlocks, hIActiveLanes));

  // execute the atomicSub kernel
  hipLaunchKernelGGL(testAtomicSub_uniValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult == verifySub(hOData, numThreads * numBlocks, hIActiveLanes));

  // execute the atomicMax kernel
  hipLaunchKernelGGL(testAtomicMax_uniValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult == verifyMax(hOData, numThreads * numBlocks, hIActiveLanes));

  // execute the atomicMin kernel
  hipLaunchKernelGGL(testAtomicMin_uniValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult == verifyMin(hOData, numThreads * numBlocks, hIActiveLanes));

  // execute the atomicAnd kernel
  hipLaunchKernelGGL(testAtomicAnd_uniValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult == verifyAnd(hOData, numThreads * numBlocks, hIActiveLanes));

  // execute the atomicOr kernel
  hipLaunchKernelGGL(testAtomicOr_uniValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult == verifyOr(hOData, numThreads * numBlocks, hIActiveLanes));

  // execute the atomicXor kernel
  hipLaunchKernelGGL(testAtomicXor_uniValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult == verifyXor(hOData, numThreads * numBlocks, hIActiveLanes));

  // Cleanup memory
  free(hOData);
  HIP_CHECK(hipFree(dOData));
  HIP_CHECK(hipFree(dIActiveLanes));
}

static void runFloatTest() {
  bool testResult = true;
  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  unsigned int numData = 4;
  unsigned int memSize = sizeof(float) * numData;
  unsigned int N = numThreads * numBlocks;

  // allocate mem for the result on host side
  float* hOData = reinterpret_cast<float*>(malloc(memSize));
  bool* hIActiveLanes = (bool*)(malloc(N * sizeof(bool)));

  // initialize the memory
  hOData[0] = Initial<float, AtomicOp::Add>::value;
  hOData[1] = Initial<float, AtomicOp::Sub>::value;
  hOData[2] = Initial<float, AtomicOp::Max>::value;
  hOData[3] = Initial<float, AtomicOp::Min>::value;

  for (unsigned int i = 0; i < N; i++) {
    // Genearte random values betwenn 0 to 9 to get random active and inactive lanes for atomic
    // operations
    hIActiveLanes[i] = std::rand() / ((RAND_MAX + 1u) / 9) % 2 == 0;
  }

  // allocate device memory for result
  float* dOData;
  bool* dIActiveLanes;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dOData), memSize));
  HIP_CHECK(hipMalloc(reinterpret_cast<bool**>(&dIActiveLanes), N * sizeof(bool)));

  // copy host memory to device to initialize to zero
  HIP_CHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dIActiveLanes, hIActiveLanes, N * sizeof(bool), hipMemcpyHostToDevice));

  // execute the atomicAdd kernel
  hipLaunchKernelGGL(testAtomicAdd_uniValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult == verifyAdd(hOData, numThreads * numBlocks, hIActiveLanes));

  // execute the atomicSub kernel
  hipLaunchKernelGGL(testAtomicSub_uniValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult == verifySub(hOData, numThreads * numBlocks, hIActiveLanes));

  // execute the atomicMax kernel
  hipLaunchKernelGGL(testAtomicMax_uniValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult == verifyMax(hOData, numThreads * numBlocks, hIActiveLanes));

  // execute the atomicMin kernel
  hipLaunchKernelGGL(testAtomicMin_uniValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult == verifyMin(hOData, numThreads * numBlocks, hIActiveLanes));

  // Cleanup memory
  free(hOData);
  free(hIActiveLanes);
  HIP_CHECK(hipFree(dOData));
  HIP_CHECK(hipFree(dIActiveLanes));
}


template <typename T> static void runDivIntTest() {
  bool testResult = true;
  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  unsigned int numData = 7;
  unsigned int memSize = sizeof(T) * numData;
  unsigned int N = numThreads * numBlocks;

  // allocate mem for the result on host side
  T* hOData = reinterpret_cast<T*>(malloc(memSize));
  T* hIDivValues = reinterpret_cast<T*>(malloc(N * sizeof(T)));
  bool* hIActiveLanes = (bool*)(malloc(N * sizeof(bool)));

  // initialize the memory
  hOData[0] = Initial<T, AtomicOp::Add>::value;
  hOData[1] = Initial<T, AtomicOp::Sub>::value;
  hOData[2] = Initial<T, AtomicOp::Max>::value;
  hOData[3] = Initial<T, AtomicOp::Min>::value;
  hOData[4] = Initial<T, AtomicOp::And>::value;
  hOData[5] = Initial<T, AtomicOp::Or>::value;
  hOData[6] = Initial<T, AtomicOp::Xor>::value;

  for (unsigned int i = 0; i < N; i++) {
    // Genearte random values betwenn 0 to 9 to get radom active and inactive lanes for atomic
    // operations
    unsigned randomValue = std::rand() / ((RAND_MAX + 1u) / 9);
    hIDivValues[i] = randomValue;
    hIActiveLanes[i] = randomValue % 2 == 0;
  }

  // allocate device memory for result
  T* dOData;
  bool* dIActiveLanes;
  T* dIDivValues;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dOData), memSize));
  HIP_CHECK(hipMalloc(reinterpret_cast<bool**>(&dIActiveLanes), N * sizeof(bool)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dIDivValues), N * sizeof(T)));

  // copy host memory to device to initialize to zero
  HIP_CHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dIActiveLanes, hIActiveLanes, N * sizeof(bool), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dIDivValues, hIDivValues, N * sizeof(T), hipMemcpyHostToDevice));

  // execute the atomicAdd kernel
  hipLaunchKernelGGL(testAtomicAdd_divValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes, dIDivValues);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult ==
          verifyAdd_divValue(hOData, numThreads * numBlocks, hIActiveLanes, hIDivValues));

  // execute the atomicSub kernel
  hipLaunchKernelGGL(testAtomicSub_divValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes, dIDivValues);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult ==
          verifySub_divValue(hOData, numThreads * numBlocks, hIActiveLanes, hIDivValues));

  // execute the atomicMax kernel
  hipLaunchKernelGGL(testAtomicMax_divValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes, dIDivValues);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult ==
          verifyMax_divValue(hOData, numThreads * numBlocks, hIActiveLanes, hIDivValues));

  // execute the atomicMin kernel
  hipLaunchKernelGGL(testAtomicMin_divValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes, dIDivValues);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult ==
          verifyMin_divValue(hOData, numThreads * numBlocks, hIActiveLanes, hIDivValues));

  // execute the atomicAnd kernel
  hipLaunchKernelGGL(testAtomicAnd_divValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes, dIDivValues);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult ==
          verifyAnd_divValue(hOData, numThreads * numBlocks, hIActiveLanes, hIDivValues));

  // execute the atomicOr kernel
  hipLaunchKernelGGL(testAtomicOr_divValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes, dIDivValues);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult ==
          verifyOr_divValue(hOData, numThreads * numBlocks, hIActiveLanes, hIDivValues));

  // execute the atomicXor kernel
  hipLaunchKernelGGL(testAtomicXor_divValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes, dIDivValues);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult ==
          verifyXor_divValue(hOData, numThreads * numBlocks, hIActiveLanes, hIDivValues));

  // Cleanup memory
  free(hOData);
  HIP_CHECK(hipFree(dOData));
  HIP_CHECK(hipFree(dIActiveLanes));
  HIP_CHECK(hipFree(dIDivValues));
}

static void runDivFloatTest() {
  bool testResult = true;
  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  unsigned int numData = 4;
  unsigned int memSize = sizeof(float) * numData;
  unsigned int N = numThreads * numBlocks;

  // allocate mem for the result on host side
  float* hOData = reinterpret_cast<float*>(malloc(memSize));
  float* hIDivValues = reinterpret_cast<float*>(malloc(N * sizeof(float)));
  bool* hIActiveLanes = (bool*)(malloc(N * sizeof(bool)));

  // initialize the memory
  hOData[0] = Initial<float, AtomicOp::Add>::value;
  hOData[1] = Initial<float, AtomicOp::Sub>::value;
  hOData[2] = Initial<float, AtomicOp::Max>::value;
  hOData[3] = Initial<float, AtomicOp::Min>::value;

  for (unsigned int i = 0; i < N; i++) {
    // Genearte random values betwenn 0 to 9 to get random active and inactive lanes for atomic
    // operations
    unsigned randomValue = std::rand() / ((RAND_MAX + 1u) / 9);
    hIDivValues[i] = (float)std::rand() / (float)randomValue;
    hIActiveLanes[i] = randomValue % 2 == 0;
  }

  // allocate device memory for result
  float* dOData;
  bool* dIActiveLanes;
  float* dIDivValues;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dOData), memSize));
  HIP_CHECK(hipMalloc(reinterpret_cast<bool**>(&dIActiveLanes), N * sizeof(bool)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dIDivValues), N * sizeof(float)));

  // copy host memory to device to initialize to zero
  HIP_CHECK(hipMemcpy(dOData, hOData, memSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dIActiveLanes, hIActiveLanes, N * sizeof(bool), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dIDivValues, hIDivValues, N * sizeof(float), hipMemcpyHostToDevice));

  // execute the atomicAdd kernel
  hipLaunchKernelGGL(testAtomicAdd_divValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes, dIDivValues);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult ==
          verifyAdd_divValue(hOData, numThreads * numBlocks, hIActiveLanes, hIDivValues));

  // execute the atomicSub kernel
  hipLaunchKernelGGL(testAtomicSub_divValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes, dIDivValues);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult ==
          verifySub_divValue(hOData, numThreads * numBlocks, hIActiveLanes, hIDivValues));

  // execute the atomicMax kernel
  hipLaunchKernelGGL(testAtomicMax_divValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes, dIDivValues);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult ==
          verifyMax_divValue(hOData, numThreads * numBlocks, hIActiveLanes, hIDivValues));

  // execute the atomicMin kernel
  hipLaunchKernelGGL(testAtomicMin_divValue, dim3(numBlocks), dim3(numThreads), 0, 0, dOData,
                     dIActiveLanes, dIDivValues);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOData, dOData, memSize, hipMemcpyDeviceToHost));

  // Compute reference solution
  REQUIRE(testResult ==
          verifyMin_divValue(hOData, numThreads * numBlocks, hIActiveLanes, hIDivValues));

  // Cleanup memory
  free(hOData);
  free(hIDivValues);
  free(hIActiveLanes);
  HIP_CHECK(hipFree(dOData));
  HIP_CHECK(hipFree(dIActiveLanes));
  HIP_CHECK(hipFree(dIDivValues));
}

/*
This testcases perform the following scenario of atomic opearations on Uniform value
for INT and UNSIGNED INT types
  // 1. atomicAdd
  // 2. atomicSub
  // 3. atomicMax
  // 4. atomicMin
  // 5. atomicAnd
  // 6. atomicOr
  // 7. atomicXor
*/
TEST_CASE("Unit_AtomicsWithRandomActiveLanesInWavefront_UniformInteger") {
  SECTION("test for int") { runIntTest<int>(); }
  SECTION("test for unsigned int") { runIntTest<unsigned int>(); }
}

/*
This testcases perform the following scenario of atomic opearations on Uniform value
for FLOAT types
  // 1. atomicAdd
  // 2. atomicSub
  // 3. atomicMax
  // 4. atomicMin
*/
TEST_CASE("Unit_AtomicsWithRandomActiveLanesInWavefront_UniformFloat") {
  SECTION("test for float") { runFloatTest(); }
}

/*
This testcases perform the following scenario of atomic opearations on Divergent values
for INT and UNSIGNED INT types
  // 1. atomicAdd
  // 2. atomicSub
  // 3. atomicMax
  // 4. atomicMin
  // 5. atomicAnd
  // 6. atomicOr
  // 7. atomicXor
*/
TEST_CASE("Unit_AtomicsWithRandomActiveLanesInWavefront_DivergentInteger") {
  SECTION("test for int") { runDivIntTest<int>(); }
  SECTION("test for unsigned int") { runDivIntTest<unsigned int>(); }
}

/*
This testcases perform the following scenario of atomic opearations on Divergent values
for FLOAT types
  // 1. atomicAdd
  // 2. atomicSub
  // 3. atomicMax
  // 4. atomicMin
*/
TEST_CASE("Unit_AtomicsWithRandomActiveLanesInWavefront_DivergentFloat") {
  SECTION("test for float") { runDivFloatTest(); }
}