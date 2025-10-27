/*
Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.

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
 * @addtogroup hipHostRegister hipHostRegister
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipHostRegister (void *hostPtr, size_t sizeBytes, unsigned int flags)` -
 * register host memory so it can be accessed from the current device.
 */

#include "hip/hip_runtime_api.h"
#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <hip_test_process.hh>

#include <utils.hh>

#define OFFSET 128
#define INITIAL_VAL 1
#define EXPECTED_VAL 2
#define ITERATION 100
#define ADDITIONAL_MEMORY_PERCENT 10

static constexpr auto LEN{1024 * 1024};
static constexpr auto LARGE_CHUNK_LEN{100 * LEN};
static constexpr auto SMALL_CHUNK_LEN{10 * LEN};

#if HT_AMD
#define TEST_SKIP(arch, msg)                                                                       \
  if (std::string::npos == arch.find("xnack+")) {                                                  \
    HipTest::HIP_SKIP_TEST(msg);                                                                   \
    return;                                                                                        \
  }
#else
#define TEST_SKIP(arch, msg)
#endif

template <typename T> __global__ void Inc(T* Ad) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  Ad[tx] = Ad[tx] + static_cast<T>(1);
}

template <typename T>
void doMemCopy(size_t numElements, int offset, T* A, T* Bh, T* Bd, bool internalRegister) {
  constexpr auto memsetval = 13.0f;
  A = A + offset;
  numElements -= offset;

  size_t sizeBytes = numElements * sizeof(T);

  if (internalRegister) {
    HIP_CHECK(hipHostRegister(A, sizeBytes, 0));
  }

  // Reset
  for (size_t i = 0; i < numElements; i++) {
    A[i] = static_cast<float>(i);
    Bh[i] = 0.0f;
  }

  HIP_CHECK(hipMemset(Bd, memsetval, sizeBytes));

  HIP_CHECK(hipMemcpy(Bd, A, sizeBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bh, Bd, sizeBytes, hipMemcpyDeviceToHost));

  // Make sure the copy worked
  ArrayMismatch(A, Bh, numElements);

  if (internalRegister) {
    HIP_CHECK(hipHostUnregister(A));
  }
}

/**
 * Test Description
 * ------------------------
 *    - This testcase verifies the hipHostRegister API by
 * 1. Allocating the memory using malloc
 * 2. hipHostRegister that variable
 * 3. Getting the corresponding device pointer of the registered varible
 * 4. Launching kernel and access the device pointer variable
 * 5. performing hipMemset on the device pointer variable
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipHostRegister_ReferenceFromKernelandhipMemset", "", int, float, double) {
  size_t sizeBytes{LEN * sizeof(TestType)};
  TestType *A, **Ad;
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  Ad = new TestType*[num_devices];
  A = reinterpret_cast<TestType*>(malloc(sizeBytes));
  SECTION("hipHostRegisterDefault") {
    HIP_CHECK(hipHostRegister(A, sizeBytes, hipHostRegisterDefault));
  }
#if (HT_AMD == 1) && (HT_LINUX == 1)
  if (!IsNavi4X()) {
    SECTION("hipExtHostRegisterUncached") {
      HIP_CHECK(hipHostRegister(A, sizeBytes, hipExtHostRegisterUncached));
    }
    SECTION("hipHostRegisterPortable | hipHostRegisterMapped | "
            "hipExtHostRegisterUncached | hipHostRegisterIoMemory") {
      HIP_CHECK(hipHostRegister(
          A, sizeBytes,
          hipHostRegisterPortable | hipHostRegisterMapped | hipExtHostRegisterUncached | 
          hipHostRegisterIoMemory));
    }
  }
#endif
  for (int i = 0; i < LEN; i++) {
    A[i] = static_cast<TestType>(1);
  }

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&Ad[i]), A, 0));
  }

  // Reference the registered device pointer Ad from inside the kernel:
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    hipLaunchKernelGGL(Inc, dim3(LEN / 32), dim3(32), 0, 0, Ad[i]);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }
  REQUIRE(A[10] == 1 + static_cast<TestType>(num_devices));
  // Reference the registered device pointer Ad in hipMemset:
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipMemset(Ad[i], 0, sizeBytes));
  }
  REQUIRE(A[10] == 0);

  HIP_CHECK(hipHostUnregister(A));

  free(A);
  delete[] Ad;
}

/**
 * Test Description
 * ------------------------
 *    - This testcase verifies that the host pointer registered by hipHostRegister API
 * is accessible from current device when xnack is on.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEMPLATE_TEST_CASE("Unit_hipHostRegister_DirectReferenceFromKernel", "", int, float, double) {
  auto flags = GENERATE(hipHostRegisterDefault, hipHostRegisterPortable, hipHostRegisterMapped);
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string arch = prop.gcnArchName;
  TEST_SKIP(arch, "Xnack+ is not supported. Skipping the test ...")
  size_t sizeBytes{LEN * sizeof(TestType)};
  TestType* A;
  A = reinterpret_cast<TestType*>(malloc(sizeBytes));
  REQUIRE(A != nullptr);
  // Initialize buffer with data
  TestType val = static_cast<TestType>(1);
  for (int i = 0; i < LEN; i++) {
    A[i] = val;
  }
  HIP_CHECK(hipHostRegister(A, sizeBytes, flags));

  // Reference the registered device pointer A from inside the kernel:
  hipLaunchKernelGGL(Inc, dim3(LEN / 32), dim3(32), 0, 0, A);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  for (int i = 0; i < LEN; i++) {
    REQUIRE(A[i] == (val + static_cast<TestType>(1)));
  }
  HIP_CHECK(hipHostUnregister(A));
  free(A);
}

/**
 * Test Description
 * ------------------------
 *    - This testcase verifies that the host pointer registered by hipHostRegister API
  is usable from multiple device when xnack is on.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEMPLATE_TEST_CASE("Unit_hipHostRegister_DirectReferenceMultGpu", "", int, float, double) {
  // 1 refers to doing hipHostRegister once for all devices
  // 0 refers to doing hipHostRegister for each device
  auto register_once = GENERATE(0, 1);
  hipDeviceProp_t prop;
  int numDevices = HipTest::getDeviceCount();
  size_t sizeBytes{LEN * sizeof(TestType)};
  TestType* A;
  A = reinterpret_cast<TestType*>(malloc(sizeBytes));
  REQUIRE(A != nullptr);
  // Register host memory only once for all device
  if (register_once == 1) {
    HIP_CHECK(hipHostRegister(A, sizeBytes, 0));
  }
  // Reference the registered device pointer A from inside all devices:
  for (int dev = 0; dev < numDevices; dev++) {
    // Initialize buffer with data
    TestType val = static_cast<TestType>(1);
    for (int i = 0; i < LEN; i++) {
      A[i] = val;
    }
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    std::string arch = prop.gcnArchName;
    if (arch.find("xnack+") == std::string::npos) {
      continue;  // Skip if xnack is not supported
    }
    // Register host memory for each device
    if (register_once == 0) {
      HIP_CHECK(hipHostRegister(A, sizeBytes, 0));
    }
    hipLaunchKernelGGL(Inc, dim3(LEN / 32), dim3(32), 0, 0, A);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    for (int i = 0; i < LEN; i++) {
      REQUIRE(A[i] == (val + static_cast<TestType>(1)));
    }
    if (register_once == 0) {
      HIP_CHECK(hipHostUnregister(A));
    }
  }
  if (register_once == 1) {
    HIP_CHECK(hipHostUnregister(A));
  }
  free(A);
}

/**
 * Test Description
 * ------------------------
 *    - This testcase verifies functionality when same host pointer is repeatedly
 * registered and unregistered.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipHostRegister_SameChunkRepeat") {
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string arch = prop.gcnArchName;
  TEST_SKIP(arch, "Xnack+ is not supported. Skipping the test ...")
  size_t sizeBytes{LEN * sizeof(uint8_t)};
  uint8_t* A;
  A = reinterpret_cast<uint8_t*>(malloc(sizeBytes));
  REQUIRE(A != nullptr);
  for (int iter = 0; iter < ITERATION; iter++) {
    // Initialize buffer with data
    memset(A, INITIAL_VAL, sizeBytes);
    HIP_CHECK(hipHostRegister(A, sizeBytes, 0));

    // Reference the registered device pointer A from inside the kernel:
    hipLaunchKernelGGL(Inc, dim3(LEN / 32), dim3(32), 0, 0, A);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    for (int i = 0; i < LEN; i++) {
      REQUIRE(A[i] == EXPECTED_VAL);
    }
    HIP_CHECK(hipHostUnregister(A));
  }
  free(A);
}

/**
 * Test Description
 * ------------------------
 *    - Allocate a large chunk of host memory. Divide the memory into smaller chunks.
 * Register each smaller chunk in one attempt. Access all the chunks in Kernel. Verify
 * results.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipHostRegister_Chunks_SingleAttempt") {
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string arch = prop.gcnArchName;
  TEST_SKIP(arch, "Xnack+ is not supported. Skipping the test ...")
  size_t sizeBytes{LARGE_CHUNK_LEN * sizeof(uint8_t)};
  size_t sizeBytesChunk{SMALL_CHUNK_LEN * sizeof(uint8_t)};
  uint8_t* A;
  A = reinterpret_cast<uint8_t*>(malloc(sizeBytes));
  REQUIRE(A != nullptr);
  // Initialize buffer with data
  memset(A, INITIAL_VAL, sizeBytes);
  uint8_t* arrPtr[LARGE_CHUNK_LEN / SMALL_CHUNK_LEN];
  for (int cnt = 0; cnt < (LARGE_CHUNK_LEN / SMALL_CHUNK_LEN); cnt++) {
    arrPtr[cnt] = A + (cnt * sizeBytesChunk);
    HIP_CHECK(hipHostRegister(arrPtr[cnt], sizeBytesChunk, 0));
  }
  // Reference each registered chunk inside the kernel:
  for (int cnt = 0; cnt < (LARGE_CHUNK_LEN / SMALL_CHUNK_LEN); cnt++) {
    uint8_t* ptrA = arrPtr[cnt];
    hipLaunchKernelGGL(Inc, dim3(SMALL_CHUNK_LEN / 32), dim3(32), 0, 0, ptrA);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    for (int i = 0; i < SMALL_CHUNK_LEN; i++) {
      REQUIRE(ptrA[i] == EXPECTED_VAL);
    }
  }
  for (int cnt = 0; cnt < (LARGE_CHUNK_LEN / SMALL_CHUNK_LEN); cnt++) {
    HIP_CHECK(hipHostUnregister(arrPtr[cnt]));
  }
  free(A);
}

/**
 * Test Description
 * ------------------------
 *    - Allocate a large chunk of host memory. Divide the memory into smaller chunks.
 * Register each smaller chunk, access the chunk in Kernel and unregister the chunk.
 * Verify results. Perform this series of operation in a round robin manner for
 * all chunks.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipHostRegister_Chunks_RoundRobin") {
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string arch = prop.gcnArchName;
  TEST_SKIP(arch, "Xnack+ is not supported. Skipping the test ...")
  size_t sizeBytes{LARGE_CHUNK_LEN * sizeof(uint8_t)};
  size_t sizeBytesChunk{SMALL_CHUNK_LEN * sizeof(uint8_t)};
  uint8_t* A;
  A = reinterpret_cast<uint8_t*>(malloc(sizeBytes));
  REQUIRE(A != nullptr);
  // Initialize buffer with data
  memset(A, INITIAL_VAL, sizeBytes);
  for (int cnt = 0; cnt < (LARGE_CHUNK_LEN / SMALL_CHUNK_LEN); cnt++) {
    uint8_t* ptrA = A + (cnt * sizeBytesChunk);
    HIP_CHECK(hipHostRegister(ptrA, sizeBytesChunk, 0));
    hipLaunchKernelGGL(Inc, dim3(SMALL_CHUNK_LEN / 32), dim3(32), 0, 0, ptrA);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    for (int i = 0; i < SMALL_CHUNK_LEN; i++) {
      REQUIRE(ptrA[i] == EXPECTED_VAL);
    }
    HIP_CHECK(hipHostUnregister(ptrA));
  }
  free(A);
}

/**
 * Test Description
 * ------------------------
 *    - This testcase verifies that the host pointer registered by hipHostRegister API
 * can be memset using hipMemset.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipHostRegister_Perform_hipMemset") {
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string arch = prop.gcnArchName;
  TEST_SKIP(arch, "Xnack+ is not supported. Skipping the test ...")
  size_t sizeBytes{LEN * sizeof(uint8_t)};
  uint8_t* A;
  A = reinterpret_cast<uint8_t*>(malloc(sizeBytes));
  REQUIRE(A != nullptr);
  // Register the host pointer
  HIP_CHECK(hipHostRegister(A, sizeBytes, 0));
  // Memset the registered pointer
  HIP_CHECK(hipMemset(A, INITIAL_VAL, sizeBytes));
  // Reference the registered device pointer A from inside the kernel:
  hipLaunchKernelGGL(Inc, dim3(LEN / 32), dim3(32), 0, 0, A);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  for (int i = 0; i < LEN; i++) {
    REQUIRE(A[i] == EXPECTED_VAL);
  }
  HIP_CHECK(hipHostUnregister(A));
  free(A);
}

/**
 * Test Description
 * ------------------------
 *    - This testcase verifies that the host pointer registered by hipHostRegister API
 * can be used with hipMemcpy.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipHostRegister_Perform_hipMemcpy") {
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string arch = prop.gcnArchName;
  TEST_SKIP(arch, "Xnack+ is not supported. Skipping the test ...")
  size_t sizeBytes{LEN * sizeof(uint8_t)};
  uint8_t *A, *B;
  A = reinterpret_cast<uint8_t*>(malloc(sizeBytes));
  REQUIRE(A != nullptr);
  B = reinterpret_cast<uint8_t*>(malloc(sizeBytes));
  REQUIRE(B != nullptr);
  memset(B, INITIAL_VAL, sizeBytes);
  // Register the host pointer
  HIP_CHECK(hipHostRegister(A, sizeBytes, 0));
  // Memcpy from B to A
  HIP_CHECK(hipMemcpy(A, B, sizeBytes, hipMemcpyDefault));
  // Reference the registered device pointer A from inside the kernel:
  hipLaunchKernelGGL(Inc, dim3(LEN / 32), dim3(32), 0, 0, A);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  // Verify if we can Memcpy from A to B
  HIP_CHECK(hipMemcpy(B, A, sizeBytes, hipMemcpyDefault));
  for (int i = 0; i < LEN; i++) {
    REQUIRE(B[i] == EXPECTED_VAL);
  }
  HIP_CHECK(hipHostUnregister(A));
  free(A);
  free(B);
}

/**
 * Test Description
 * ------------------------
 *    - Oversubscription: This testcase allocates host memory of size > total
 * GPU memory. Register the memory and try accessing it from kernel. Verify
 * the behaviour.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipHostRegister_Oversubscription") {
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string arch = prop.gcnArchName;
  TEST_SKIP(arch, "Xnack+ is not supported. Skipping the test ...")
  size_t maxGpuMem = 0, availableMem = 0;
  // Get available GPU memory and total GPU memory
  HIP_CHECK(hipMemGetInfo(&availableMem, &maxGpuMem));
  size_t allocsize = maxGpuMem + ((maxGpuMem * ADDITIONAL_MEMORY_PERCENT) / 100);
  // Get free host In bytes
  size_t hostMemFree = HipTest::getMemoryAmount() * 1024 * 1024;
  // Ensure that allocsize < hostMemFree
  if (allocsize >= hostMemFree) {
    HipTest::HIP_SKIP_TEST("Available Host Memory is not sufficient ...");
    return;
  }
  uint8_t* A = reinterpret_cast<uint8_t*>(malloc(allocsize));
  REQUIRE(A != nullptr);
  size_t used_size = LEN;
  // Inititalize only the first used_size bytes chunk
  memset(A, INITIAL_VAL, used_size);
  // Inititalize only the last used_size bytes chunk
  memset((A + allocsize - used_size), INITIAL_VAL, used_size);
  // Register the entire host memory chunk
  HIP_CHECK(hipHostRegister(A, allocsize, 0));
  // Reference only the first used_size bytes
  hipLaunchKernelGGL(Inc, dim3(used_size / 32), dim3(32), 0, 0, A);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  for (int i = 0; i < used_size; i++) {
    REQUIRE(A[i] == EXPECTED_VAL);
  }
  // Reference only the last used_size bytes chunk
  uint8_t* B = (A + allocsize - used_size);
  hipLaunchKernelGGL(Inc, dim3(used_size / 32), dim3(32), 0, 0, B);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  for (int i = 0; i < used_size; i++) {
    REQUIRE(B[i] == EXPECTED_VAL);
  }
  HIP_CHECK(hipHostUnregister(A));
  free(A);
}

/**
 * Test Description
 * ------------------------
 *    - This testcase verifies that the host pointer registered by hipHostRegister API
 * can be used with Async APIs (hipMemsetAsync, hipMemcpyAsync and kernel) on a user
 * defined stream.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipHostRegister_AsyncApis") {
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string arch = prop.gcnArchName;
  bool useRegPtrInDev = false;
#if HT_AMD
  if (std::string::npos == arch.find("xnack+")) {
    useRegPtrInDev = false;
  } else {
    useRegPtrInDev = true;
  }
#else
  useRegPtrInDev = GENERATE(true, false);
#endif
  size_t sizeBytes{LEN * sizeof(uint32_t)};
  uint32_t *A, *B, *dPtr;
  A = reinterpret_cast<uint32_t*>(malloc(sizeBytes));
  REQUIRE(A != nullptr);
  B = reinterpret_cast<uint32_t*>(malloc(sizeBytes));
  REQUIRE(B != nullptr);
  for (int i = 0; i < LEN; i++) {
    B[i] = i;
  }
  // Register the host pointer
  HIP_CHECK(hipHostRegister(A, sizeBytes, 0));
  if (useRegPtrInDev) {
    dPtr = A;
  } else {
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), A, 0));
  }
  hipStream_t strm{nullptr};
  HIP_CHECK(hipStreamCreate(&strm));
  // Memcpy from B to A
  HIP_CHECK(hipMemcpyAsync(dPtr, B, sizeBytes, hipMemcpyHostToDevice, strm));
  // Reference the registered device pointer A from inside the kernel:
  hipLaunchKernelGGL(Inc, dim3(LEN / 32), dim3(32), 0, strm, dPtr);
  HIP_CHECK(hipMemcpyAsync(B, dPtr, sizeBytes, hipMemcpyDeviceToHost, strm));
  HIP_CHECK(hipStreamSynchronize(strm));
  for (int i = 0; i < LEN; i++) {
    REQUIRE(B[i] == (i + 1));
  }
  HIP_CHECK(hipStreamDestroy(strm));
  HIP_CHECK(hipHostUnregister(A));
  free(A);
  free(B);
}

/**
 * Test Description
 * ------------------------
 *    - This testcase verifies the behaviour of host registered memory when
 * used with hipGraph.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipHostRegister_Graphs") {
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string arch = prop.gcnArchName;
  bool useRegPtrInDev = false;
#if HT_AMD
  if (std::string::npos == arch.find("xnack+")) {
    useRegPtrInDev = false;
  } else {
    useRegPtrInDev = true;
  }
#else
  useRegPtrInDev = GENERATE(true, false);
#endif
  size_t sizeBytes{LEN * sizeof(uint32_t)};
  uint32_t *A, *B, *dPtr;
  A = reinterpret_cast<uint32_t*>(malloc(sizeBytes));
  REQUIRE(A != nullptr);
  B = reinterpret_cast<uint32_t*>(malloc(sizeBytes));
  REQUIRE(B != nullptr);
  for (int i = 0; i < LEN; i++) {
    B[i] = i;
  }
  // Register the host pointer
  HIP_CHECK(hipHostRegister(A, sizeBytes, 0));
  if (useRegPtrInDev) {
    dPtr = A;
  } else {
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), A, 0));
  }
  // Use dPtr in graphs
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D, memcpyD2H;
  hipGraphNode_t kernel_vecInc;
  void* kernelArgs1[] = {&dPtr};
  hipKernelNodeParams kernelNodeParams{};
  kernelNodeParams.func = reinterpret_cast<void*>(Inc<uint32_t>);
  kernelNodeParams.gridDim = dim3(LEN / 32);
  kernelNodeParams.blockDim = dim3(32);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecInc, graph, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D, graph, nullptr, 0, dPtr, B, sizeBytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H, graph, nullptr, 0, B, dPtr, sizeBytes,
                                    hipMemcpyDeviceToHost));
  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D, &kernel_vecInc, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecInc, &memcpyD2H, 1));
  // Instantiate and execute Graph
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  // Verify Result
  for (int i = 0; i < LEN; i++) {
    REQUIRE(B[i] == (i + 1));
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipHostUnregister(A));
  free(A);
  free(B);
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 *    - This testcase measures performance when same memory chunk is repeatedly
 * registered and unregistered.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipHostRegister_RegUnreg_Perf_SameChunk") {
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string arch = prop.gcnArchName;
  if (std::string::npos == arch.find("xnack+")) {
    HipTest::HIP_SKIP_TEST("Xnack+ is not supported. Skipping the test ...");
    return;
  }
  hip::SpawnProc proc("hipHostRegisterPerf", true);
  REQUIRE(proc.run("svm_enable 1") == 0);
  float perf_svm_enable = std::stof(proc.getOutput());
  INFO("perf_svm_enable: " << perf_svm_enable);
  REQUIRE(proc.run("svm_disable 1") == 0);
  float perf_svm_disable = std::stof(proc.getOutput());
  INFO("perf_svm_disable: " << perf_svm_disable);
  REQUIRE(perf_svm_enable <= perf_svm_disable);
}

/**
 * Test Description
 * ------------------------
 *    - This testcase measures performance when different memory chunks
 * are repeatedly registered and unregistered.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipHostRegister_RegUnreg_Perf_DiffChunk") {
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string arch = prop.gcnArchName;
  if (std::string::npos == arch.find("xnack+")) {
    HipTest::HIP_SKIP_TEST("Xnack+ is not supported. Skipping the test ...");
    return;
  }
  hip::SpawnProc proc("hipHostRegisterPerf", true);
  REQUIRE(proc.run("svm_enable 0") == 0);
  float perf_svm_enable = std::stof(proc.getOutput());
  INFO("perf_svm_enable: " << perf_svm_enable);
  REQUIRE(proc.run("svm_disable 0") == 0);
  float perf_svm_disable = std::stof(proc.getOutput());
  INFO("perf_svm_disable: " << perf_svm_disable);
  REQUIRE(perf_svm_enable <= perf_svm_disable);
}

/**
 * Test Description
 * ------------------------
 *    - This testcase measures performance when same memory chunk is repeatedly
 * registered and unregistered on multiple GPUs.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipHostRegister_RegUnreg_Perf_SameChunk_MGPU") {
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string arch = prop.gcnArchName;
  if (std::string::npos == arch.find("xnack+")) {
    HipTest::HIP_SKIP_TEST("Xnack+ is not supported. Skipping the test ...");
    return;
  }
  int dev_count = HipTest::getDeviceCount();
  if (dev_count < 2) {
    HipTest::HIP_SKIP_TEST("Only 1 GPU available. Skipping this test ...");
    return;
  }
  hip::SpawnProc proc("hipHostRegisterPerf", true);
  REQUIRE(proc.run("svm_enable 2") == 0);
  float perf_svm_enable = std::stof(proc.getOutput());
  INFO("perf_svm_enable: " << perf_svm_enable);
  REQUIRE(proc.run("svm_disable 2") == 0);
  float perf_svm_disable = std::stof(proc.getOutput());
  INFO("perf_svm_disable: " << perf_svm_disable);
  REQUIRE(perf_svm_enable <= perf_svm_disable);
}

/**
 * Test Description
 * ------------------------
 *    - This testcase verifies whether hipMemAdvise can be used with
 * host memory registered with hipHostRegister.
 * registered and unregistered.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipHostRegister_MemAdvise_SetGet") {
  // Execute the test only if xnack is supported
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string arch = prop.gcnArchName;
  if ((std::string::npos == arch.find("xnack+")) || (prop.concurrentManagedAccess == 0)) {
    const char* msg = "Xnack/ConcurrentAccess not supported. Skipping test";
    HipTest::HIP_SKIP_TEST(msg);
    return;
  }
  int numDevices = HipTest::getDeviceCount();
  size_t sizeBytes{LEN * sizeof(uint8_t)};
  uint8_t* A;
  A = reinterpret_cast<uint8_t*>(malloc(sizeBytes));
  REQUIRE(A != nullptr);
  memset(A, INITIAL_VAL, sizeBytes);
  HIP_CHECK(hipHostRegister(A, sizeBytes, 0));
  int out = 0;
  SECTION("Attribute = hipMemAdviseSetReadMostly") {
    HIP_CHECK(hipMemAdvise(A, sizeBytes, hipMemAdviseSetReadMostly, 0));
    HIP_CHECK(hipMemRangeGetAttribute(&out, 4, hipMemRangeAttributeReadMostly, A, sizeBytes));
    REQUIRE(out == 1);
    HIP_CHECK(hipMemAdvise(A, sizeBytes, hipMemAdviseUnsetReadMostly, 0));
    HIP_CHECK(hipMemRangeGetAttribute(&out, 4, hipMemRangeAttributeReadMostly, A, sizeBytes));
    REQUIRE(out == 0);
  }
  SECTION("Attribute = hipMemAdviseSetPreferredLocation") {
    HIP_CHECK(hipMemAdvise(A, sizeBytes, hipMemAdviseSetPreferredLocation, hipCpuDeviceId));
    HIP_CHECK(hipMemRangeGetAttribute(&out, sizeof(int), hipMemRangeAttributePreferredLocation, A,
                                      sizeBytes));
    REQUIRE(out == hipCpuDeviceId);
    for (int dev = 0; dev < numDevices; dev++) {
      HIP_CHECK(hipMemAdvise(A, sizeBytes, hipMemAdviseSetPreferredLocation, dev));
      HIP_CHECK(hipMemRangeGetAttribute(&out, sizeof(int), hipMemRangeAttributePreferredLocation, A,
                                        sizeBytes));
      REQUIRE(out == dev);
    }
    HIP_CHECK(hipMemAdvise(A, sizeBytes, hipMemAdviseUnsetPreferredLocation, 0));
    HIP_CHECK(hipMemRangeGetAttribute(&out, sizeof(int), hipMemRangeAttributePreferredLocation, A,
                                      sizeBytes));
    REQUIRE(out == hipInvalidDeviceId);
  }
  SECTION("Attribute = hipMemAdviseSetAccessedBy") {
    size_t size = numDevices * sizeof(int);
    int* chkOut = reinterpret_cast<int*>(malloc(size));
    HIP_CHECK(hipMemAdvise(A, sizeBytes, hipMemAdviseSetAccessedBy, hipCpuDeviceId));
    for (int dev = 0; dev < numDevices; dev++) {
      HIP_CHECK(hipMemAdvise(A, sizeBytes, hipMemAdviseSetAccessedBy, dev));
    }
    HIP_CHECK(hipMemRangeGetAttribute(chkOut, size, hipMemRangeAttributeAccessedBy, A, sizeBytes));
    for (int dev = 0; dev < numDevices; dev++) {
      REQUIRE(chkOut[dev] == dev);
    }
    free(chkOut);
  }
  HIP_CHECK(hipHostUnregister(A));
  free(A);
}
#endif
/**
 * Test Description
 * ------------------------
 *    - This testcase verifies hipHostRegister API by performing memcpy
 * on the hipHostRegistered variable.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipHostRegister_Memcpy", "", int, float, double) {
  // 1 refers to hipHostRegister
  // 0 refers to malloc
  auto mem_type = GENERATE(0, 1);
  HIP_CHECK(hipSetDevice(0));

  size_t sizeBytes = LEN * sizeof(TestType);
  TestType* A = reinterpret_cast<TestType*>(malloc(sizeBytes));

  // Copy to B, this should be optimal pinned malloc copy:
  // Note we are using the host pointer here:
  TestType *Bh, *Bd;
  Bh = reinterpret_cast<TestType*>(malloc(sizeBytes));
  HIP_CHECK(hipMalloc(&Bd, sizeBytes));

  REQUIRE(LEN > OFFSET);
  if (mem_type) {
    for (size_t i = 0; i < OFFSET; i++) {
      doMemCopy<TestType>(LEN, i, A, Bh, Bd, true /*internalRegister*/);
    }
  } else {
    HIP_CHECK(hipHostRegister(A, sizeBytes, 0));
    for (size_t i = 0; i < OFFSET; i++) {
      doMemCopy<TestType>(LEN, i, A, Bh, Bd, false /*internalRegister*/);
    }
    HIP_CHECK(hipHostUnregister(A));
  }

  free(A);
  free(Bh);
  HIP_CHECK(hipFree(Bd));
}

template <typename T> __global__ void fill_kernel(T* dataPtr, T value) {
  size_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  dataPtr[tid] = value;
}

/**
 * Test Description
 * ------------------------
 *    - This testcase verifies all the supported flags of hipHostRegister.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipHostRegister_Flags", "", int, float, double) {
  size_t sizeBytes = 1 * sizeof(TestType);
  TestType* hostPtr = reinterpret_cast<TestType*>(malloc(sizeBytes));

  /* Flags aren't used for AMD devices currently */
  struct FlagType {
    unsigned int value;
    bool valid;
  };

  /* EXSWCPHIPT-29 - 0x08 is hipHostRegisterReadOnly which currently doesn't
  have a definition in the headers */
  /* hipHostRegisterIoMemory is a valid flag but requires access to I/O mapped
  memory to be tested */
  FlagType flags = GENERATE(
      FlagType{hipHostRegisterDefault, true}, FlagType{hipHostRegisterPortable, true},
      FlagType{0x08, true}, FlagType{hipHostRegisterPortable | hipHostRegisterMapped, true},
      FlagType{hipHostRegisterPortable | hipHostRegisterMapped | 0x08, true},
#if (HT_AMD == 1) && (HT_LINUX == 1)
      FlagType{hipHostRegisterIoMemory, true},
      FlagType{hipExtHostRegisterUncached, true},
      FlagType{hipHostRegisterPortable | hipHostRegisterMapped | hipExtHostRegisterUncached, true},
#endif
      FlagType{0xF0, false}, FlagType{0xFFF2, false}, FlagType{0xFFFFFFFF, false});

#if (HT_AMD == 1) && (HT_LINUX == 1)
  if (IsNavi4X() && (flags.value & hipExtHostRegisterUncached)) {
    return;
  }
#endif
  INFO("Testing hipHostRegister flag: " << flags.value);
  if (flags.valid) {
    HIP_CHECK(hipHostRegister(hostPtr, sizeBytes, flags.value));
    HIP_CHECK(hipHostUnregister(hostPtr));
  } else {
    HIP_CHECK_ERROR(hipHostRegister(hostPtr, sizeBytes, flags.value), hipErrorInvalidValue);
  }
  free(hostPtr);
}
/**
 * Test Description
 * ------------------------
 *    - These negative tests checks invalid parameter values.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipHostRegister_Negative", "", int, float, double) {
  TestType* hostPtr = nullptr;

  size_t sizeBytes = 1 * sizeof(TestType);
  SECTION("hipHostRegister Negative Test - nullptr") {
    HIP_CHECK_ERROR(hipHostRegister(hostPtr, 1, 0), hipErrorInvalidValue);
  }

  hostPtr = reinterpret_cast<TestType*>(malloc(sizeBytes));
  SECTION("hipHostRegister Negative Test - zero size") {
    HIP_CHECK_ERROR(hipHostRegister(hostPtr, 0, 0), hipErrorInvalidValue);
  }

  size_t devMemAvail{0}, devMemFree{0};
  HIP_CHECK(hipMemGetInfo(&devMemFree, &devMemAvail));
  auto hostMemFree = HipTest::getMemoryAmount() /* In MB */ * 1024 * 1024;  // In bytes
  REQUIRE(devMemFree > 0);
  REQUIRE(devMemAvail > 0);
  REQUIRE(hostMemFree > 0);

  // which is the limiter cpu or gpu
  size_t memFree = (std::max)(devMemFree, hostMemFree);

  SECTION("hipHostRegister Negative Test - invalid memory size") {
    HIP_CHECK_ERROR(hipHostRegister(hostPtr, memFree, 0), hipErrorInvalidValue);
  }

  free(hostPtr);
  SECTION("hipHostRegister Negative Test - freed memory") {
    HIP_CHECK_ERROR(hipHostRegister(hostPtr, 0, 0), hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipHostRegister_Capture") {
  constexpr size_t kBufferSize = 1024;
  auto buffer = std::make_unique<int[]>(kBufferSize);
  hipError_t capture_error = hipSuccess;

  constexpr bool kRelaxedModeAllowed = true;
  BEGIN_CAPTURE_SYNC(capture_error, kRelaxedModeAllowed);
  HIP_CHECK_ERROR(hipHostRegister(buffer.get(), kBufferSize, 0), capture_error);
  END_CAPTURE_SYNC(capture_error);

  if (capture_error == hipSuccess) {
    HIP_CHECK(hipHostUnregister(buffer.get()));
  }
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
