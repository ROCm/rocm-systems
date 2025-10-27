
/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef _WIN32
#define NOMINMAX
#endif /* _WIN32 */

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_context.hh>
#include <hip_test_helper.hh>
#include <kernels.hh>

#define MEMORY_PERCENT 10
#define BLOCK_SIZE 512
#define VALUE 32

enum SYNC_VALUES { SYNC_EVENT, SYNC_STREAM, SYNC_DEVICE };

static constexpr int NUMELEMENTS{1024 * 16};
static constexpr size_t SIZEBYTES{NUMELEMENTS * sizeof(int)};
static std::vector<std::string> syncMsg = {"event", "stream", "device"};

static void CheckHostPointer(int NUMELEMENTS, int* ptr, unsigned eventFlags, int syncMethod,
                             std::string msg) {
  INFO("test: CheckHostPointer "
       << msg << " eventFlags = " << std::hex << eventFlags
       << ((eventFlags & hipEventReleaseToDevice) ? " hipEventReleaseToDevice" : "")
       << ((eventFlags & hipEventReleaseToSystem) ? " hipEventReleaseToSystem" : "")
       << " ptr=" << ptr << " syncMethod=" << syncMsg[syncMethod] << "\n");

  hipStream_t s;
  hipEvent_t e;

  // Init:
  HIP_CHECK(hipStreamCreate(&s));
  HIP_CHECK(hipEventCreateWithFlags(&e, eventFlags))
  dim3 dimBlock(64, 1, 1);
  dim3 dimGrid(NUMELEMENTS / dimBlock.x, 1, 1);

  const int expected = 13;

  // Init array to know state:
  HipTest::launchKernel(Set, dimGrid, dimBlock, 0, 0x0, ptr, -42);
  HIP_CHECK(hipDeviceSynchronize());

  HipTest::launchKernel(Set, dimGrid, dimBlock, 0, s, ptr, expected);
  HIP_CHECK(hipEventRecord(e, s));

  // Host waits for event :
  switch (syncMethod) {
    case SYNC_EVENT:
      HIP_CHECK(hipEventSynchronize(e));
      break;
    case SYNC_STREAM:
      HIP_CHECK(hipStreamSynchronize(s));
      break;
    case SYNC_DEVICE:
      HIP_CHECK(hipDeviceSynchronize());
      break;
    default:
      REQUIRE(false);
  }

  for (int i = 0; i < NUMELEMENTS; i++) {
    INFO("mismatch at index:" << i << "Got value:" << ptr[i] << "Expected value :" << expected
                              << "\n");
    REQUIRE(ptr[i] == expected);
  }

  HIP_CHECK(hipStreamDestroy(s));
  HIP_CHECK(hipEventDestroy(e));
}

static __global__ void write_integer(int* memory, int value) {
  if (memory) {
    *memory = value;
  }
}

int get_flags() {
  return GENERATE(hipHostMallocDefault, hipHostMallocPortable, hipHostMallocMapped,
                  hipHostMallocWriteCombined, hipHostMallocPortable | hipHostMallocMapped,
                  hipHostMallocPortable | hipHostMallocWriteCombined,
                  hipHostMallocMapped | hipHostMallocWriteCombined,
                  hipHostMallocPortable | hipHostMallocMapped | hipHostMallocWriteCombined);
}

TEST_CASE("Unit_hipHostAlloc_Positive") {
  int* host_memory = nullptr;
  int flags = get_flags();

  HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&host_memory), sizeof(int), flags));

  REQUIRE(host_memory != nullptr);

  HIP_CHECK(hipFreeHost(host_memory));
}

TEST_CASE("Unit_hipHostAlloc_DataValidation") {
  int validation_number = 10;
  int* host_memory = nullptr;
  int* device_memory = nullptr;
  hipEvent_t event = nullptr;
  int flags = get_flags();

  HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&host_memory), sizeof(int), flags));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_memory), host_memory, 0));

  write_integer<<<1, 1>>>(device_memory, validation_number);

  SECTION("device sync") { HIP_CHECK(hipDeviceSynchronize()); }

  SECTION("event sync") {
    HIP_CHECK(hipEventCreateWithFlags(&event, 0));
    HIP_CHECK(hipEventRecord(event, nullptr));
    HIP_CHECK(hipEventSynchronize(event));
  }

  SECTION("stream sync") { HIP_CHECK(hipStreamSynchronize(nullptr)); }

  REQUIRE(*host_memory == validation_number);

  if (event != nullptr) {
    HIP_CHECK(hipEventDestroy(event));
  }

  HIP_CHECK(hipFreeHost(host_memory));
}

TEST_CASE("Unit_hipHostAlloc_Negative") {
  int* host_memory = nullptr;
  int flags = get_flags();

  SECTION("host memory is nullptr") {
    HIP_CHECK_ERROR(hipHostAlloc(nullptr, sizeof(int), flags), hipErrorInvalidValue);
  }

  SECTION("size is negative") {
    HIP_CHECK_ERROR(hipHostAlloc(reinterpret_cast<void**>(&host_memory), -1, flags),
                    hipErrorOutOfMemory);
  }

  SECTION("flag is out of range") {
    unsigned int flag = 999;
    HIP_CHECK_ERROR(hipHostAlloc(reinterpret_cast<void**>(&host_memory), sizeof(int), flag),
                    hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This testcase performs the basic scenario of hipHostAlloc API by:
 *    Allocates the memory using hipHostAlloc API.
 *    Launches the kernel and performs vector addition.
 *    Validates the result.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipHostAlloc_Basic") {
  static constexpr auto LEN{1024 * 1024};
  static constexpr auto SIZE{LEN * sizeof(float)};

  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  if (prop.canMapHostMemory != 1) {
    SUCCEED("Doesn't support HostPinned Memory");
  } else {
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    unsigned int flag = 0;
    HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&A_h), SIZE,
                           hipHostAllocWriteCombined | hipHostAllocMapped));

    SECTION("hipHostAllocDefault") { flag = hipHostAllocDefault; }
#if (HT_AMD == 1) && (HT_LINUX == 1)
    if (!IsNavi4X()) {
      SECTION("hipHostAllocUncached") { flag = hipHostAllocUncached; }
    }
#endif

    HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&B_h), SIZE, flag));
    HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&C_h), SIZE, hipHostAllocMapped));

    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), A_h, 0));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&C_d), C_h, 0));

    HipTest::setDefaultData<float>(LEN, A_h, B_h, C_h);

    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&B_d), SIZE));
    HIP_CHECK(hipMemcpy(B_d, B_h, SIZE, hipMemcpyHostToDevice));

    dim3 dimGrid(LEN / 512, 1, 1);
    dim3 dimBlock(512, 1, 1);
    HipTest::launchKernel<float>(HipTest::vectorADD<float>, dimGrid, dimBlock, 0, 0,
                                 static_cast<const float*>(A_d), static_cast<const float*>(B_d),
                                 C_d, static_cast<size_t>(LEN));
    HIP_CHECK(hipMemcpy(C_h, C_d, LEN * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HipTest::checkVectorADD<float>(A_h, B_h, C_h, NUMELEMENTS);

    HIP_CHECK(hipHostFree(A_h));
    HIP_CHECK(hipHostFree(B_h));
    HIP_CHECK(hipHostFree(C_h));
    HIP_CHECK(hipFree(B_d));
  }
}

/**
 * This testcase verifies the hipHostAlloc API by allocating memory
 * using default flag-
 * Launches the kernel and modifies the variable
 * using different synchronization techniquies
 * validates the result.
 */
TEST_CASE("Unit_hipHostAlloc_Default") {
  int* A = nullptr;
  HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&A), SIZEBYTES, hipHostMallocDefault));
  std::string kPtrType{"default"};
  CheckHostPointer(NUMELEMENTS, A, 0, SYNC_DEVICE, kPtrType);
  CheckHostPointer(NUMELEMENTS, A, 0, SYNC_STREAM, kPtrType);
  CheckHostPointer(NUMELEMENTS, A, 0, SYNC_EVENT, kPtrType);
  HIP_CHECK(hipHostFree(A));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase verifies the hipHostAlloc API by:
 *  Allocating memory using hipHostMallocNonCoherent flag.
 *  This is a negative test as hipHostMallocNonCoherent
 *  flag is not supported by hipHostAlloc.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
#if HT_AMD
TEST_CASE("Unit_hipHostAlloc_Negative_NonCoherent") {
  int* A = nullptr;
  REQUIRE(hipHostAlloc(reinterpret_cast<void**>(&A), SIZEBYTES, hipHostMallocNonCoherent) ==
          hipErrorInvalidValue);
  REQUIRE(A == nullptr);
}
#endif

/**
 * Test Description
 * ------------------------
 *  - This testcase verifies the hipHostAlloc API by:
 *  Allocating memory using hipHostMallocCoherent flag.
 *  This is a negative test as hipHostMallocCoherent
 *  flag is not supported by hipHostAlloc.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
#if HT_AMD
TEST_CASE("Unit_hipHostAlloc_Negative_Coherent") {
  int* A = nullptr;
  REQUIRE(hipHostAlloc(reinterpret_cast<void**>(&A), SIZEBYTES, hipHostMallocCoherent) ==
          hipErrorInvalidValue);
  REQUIRE(A == nullptr);
}
#endif

/**
 * Test Description
 * ------------------------
 *  - This testcase verifies the hipHostAlloc API by:
 *  Allocating memory using hipHostMallocNumaUser flag.
 *  This is a negative test as hipHostMallocNumaUser
 *  flag is not supported by hipHostAlloc.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
#if HT_AMD
TEST_CASE("Unit_hipHostAlloc_Negative_NumaUser") {
  int* A = nullptr;
  REQUIRE(hipHostAlloc(reinterpret_cast<void**>(&A), SIZEBYTES, hipHostMallocNumaUser) ==
          hipErrorInvalidValue);
  REQUIRE(A == nullptr);
}
#endif

/**
 * Test Description
 * ------------------------
 *  - This testcase verifies the hipHostAlloc API by:
 *  Allocating more memory than total GPU memory.
 *  Validate return hipSuccess.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipHostAlloc_AllocateMoreThanAvailGPUMemory") {
  char* A = nullptr;
  size_t maxGpuMem = 0, availableMem = 0;
  // Get available GPU memory and total GPU memory
  HIP_CHECK(hipMemGetInfo(&availableMem, &maxGpuMem));
#if defined(_WIN32)
  size_t allocsize = availableMem - (256 * 1024 * 1024);
  allocsize -= allocsize * (MEMORY_PERCENT / 100.0);
#else
  size_t allocsize = maxGpuMem + ((maxGpuMem * MEMORY_PERCENT) / 100);
#endif
  // Get free host In bytes
  size_t hostMemFree = HipTest::getMemoryAmount() * 1024 * 1024;
  // Ensure that allocsize < hostMemFree
  if (allocsize < hostMemFree) {
    HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&A), allocsize, hipHostMallocDefault));
    HIP_CHECK(hipHostFree(A));
  } else {
    WARN("Skipping test as CPU memory is less than GPU memory");
  }
}

/**
 * Test Description
 * ------------------------
 *  - This testcase verifies the hipHostAlloc API by:
 *  Test hipHostAlloc() api with ptr as nullptr and check for return value.
 *  Test hipHostAlloc() api with size as max(size_t) and check for OOM error.
 *  Pass size as zero for hipHostAlloc() api and check ptr is reset with
 *  with return value success.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipHostAlloc_ArgValidation") {
  constexpr size_t allocSize = 1000;
  char* ptr;

  SECTION("Pass ptr as nullptr") {
    HIP_CHECK_ERROR(hipHostAlloc(static_cast<void**>(nullptr), allocSize, hipHostMallocDefault),
                    hipErrorInvalidValue);
  }

  SECTION("Size as max(size_t)") {
    HIP_CHECK_ERROR(hipHostAlloc(reinterpret_cast<void**>(&ptr),
                                 (std::numeric_limits<std::size_t>::max()), hipHostMallocDefault),
                    hipErrorMemoryAllocation);
  }

  SECTION("Pass size as zero and check ptr reset") {
    HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&ptr), 0, hipHostMallocDefault));
    REQUIRE(ptr == nullptr);
  }
}

TEST_CASE("Unit_hipHostAlloc_Capture") {
  int* host_memory = nullptr;
  int flags = get_flags();

  hipError_t capture_error = hipSuccess;
  constexpr bool kRelaxedModeAllowed = true;
  BEGIN_CAPTURE_SYNC(capture_error, kRelaxedModeAllowed);

  HIP_CHECK_ERROR(hipHostAlloc(reinterpret_cast<void**>(&host_memory), sizeof(int), flags),
                  capture_error);

  END_CAPTURE_SYNC(capture_error);

  if (capture_error == hipSuccess) {
    REQUIRE(host_memory != nullptr);
    HIP_CHECK(hipFreeHost(host_memory));
  }
}
