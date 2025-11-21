/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <utils.hh>

#if __linx__
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "hip_vmm_common.hh"
#define THREADS_PER_BLOCK 512

/**
 * Kernel to perform Square of input data.
 */
static __global__ void squareKernel(int* Buff) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int temp = Buff[i] * Buff[i];
  Buff[i] = temp;
}

/**
 * Helper function to get the granularity of the device
 */
size_t GetGranularity(hipDevice_t device) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  assert(granularity > 0);
  return granularity;
}

/**
 * Helper function to create the Physical memory of given size
 */
hipMemGenericAllocationHandle_t GetPhysicalMemory(hipDevice_t device, size_t size_mem) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  return handle;
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following Negative scenarios,
 *  - 1) With device pointer as nullptr
 *  - 2) With size as 0
 *  - 3) With Invalid hipMemRangeHandleType
 *  - 4) With Invalid Flags
 *  - 5) With device pointer as already freed memory
 *  - 6) With Host Memory
 *  - 7) With Unmapped Virtual memory
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_Negative") {
  int handle = -1;
  int* dptr = nullptr;
  constexpr int size = 10;
  constexpr int sizeBytes = size * sizeof(int);
  HIP_CHECK(hipMalloc(&dptr, sizeBytes));

  SECTION("nullptr") {
    HIP_CHECK_ERROR(hipMemGetHandleForAddressRange(&handle, nullptr, sizeBytes,
                                                   hipMemRangeHandleTypeDmaBufFd, 0),
                    hipErrorInvalidValue);
  }

  SECTION("size 0") {
    HIP_CHECK_ERROR(
        hipMemGetHandleForAddressRange(&handle, dptr, 0, hipMemRangeHandleTypeDmaBufFd, 0),
        hipErrorInvalidValue);
  }

  SECTION("Invalid Handle type") {
    HIP_CHECK_ERROR(hipMemGetHandleForAddressRange(&handle, dptr, sizeBytes,
                                                   static_cast<hipMemRangeHandleType>(-1), 0),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid Flags") {
    HIP_CHECK_ERROR(hipMemGetHandleForAddressRange(&handle, dptr, sizeBytes,
                                                   hipMemRangeHandleTypeDmaBufFd, 0xFF),
                    hipErrorInvalidValue);
  }

  SECTION("With Freed Memory") {
    int* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, sizeBytes));
    HIP_CHECK(hipFree(devMem));

    HIP_CHECK_ERROR(hipMemGetHandleForAddressRange(&handle, devMem, sizeBytes,
                                                   hipMemRangeHandleTypeDmaBufFd, 0),
                    hipErrorInvalidValue);
  }

  SECTION("With Host memory") {
    int* hptr = new int[size];
    HIP_CHECK_ERROR(
        hipMemGetHandleForAddressRange(&handle, hptr, sizeBytes, hipMemRangeHandleTypeDmaBufFd, 0),
        hipErrorInvalidValue);
    delete[] hptr;
  }

  SECTION("With Unmapped Virtual Memory") {
    hipDevice_t device;
    constexpr int kDeviceId = 0;
    HIP_CHECK(hipDeviceGet(&device, kDeviceId));
    checkVMMSupported(device);

    size_t granularity = GetGranularity(kDeviceId);
    assert(granularity > 0);

    size_t size_mem = ((granularity + sizeBytes - 1) / granularity) * granularity;
    hipDeviceptr_t ptrA = nullptr;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, granularity, 0, 0));
    REQUIRE(ptrA != nullptr);

    HIP_CHECK_ERROR(
        hipMemGetHandleForAddressRange(&handle, ptrA, size_mem, hipMemRangeHandleTypeDmaBufFd, 0),
        hipErrorInvalidValue);
  }

  HIP_CHECK(hipFree(dptr));
}

/**
 * Helper function to create a device memory, fills the data and
 * returns a devie memory pointer
 */
void* createDeviceMemoryAndFillData(int size) {
  int sizeBytes = size * sizeof(int);
  void* srcDevMem = nullptr;
  HIP_CHECK(hipMalloc(&srcDevMem, sizeBytes));
  REQUIRE(srcDevMem != nullptr);

  int* srcHostMem = nullptr;
  srcHostMem = reinterpret_cast<int*>(malloc(sizeBytes));
  REQUIRE(srcHostMem != nullptr);
  for (int i = 0; i < size; i++) {
    srcHostMem[i] = i;
  }

  HIP_CHECK(hipMemcpy(srcDevMem, srcHostMem, sizeBytes, hipMemcpyHostToDevice));

  free(srcHostMem);
  return srcDevMem;
}

/**
 * Helper function to create a virtual memory, fills the data and
 * returns a devie pointer
 */
hipDeviceptr_t createVirtualMemoryAndFillData(int size, int* reservedAddrSize, int device = 0) {
  size_t granularity = GetGranularity(device);
  if (granularity <= 0) {
    std::cout << "Invalid Granularity" << std::endl;
    return nullptr;
  }

  int* srcHostMem = reinterpret_cast<int*>(malloc(size * sizeof(int)));
  for (int i = 0; i < size; i++) {
    srcHostMem[i] = i;
  }

  size_t size_mem = ((granularity + (size * sizeof(int)) - 1) / granularity) * granularity;
  hipDeviceptr_t ptrA = nullptr;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, granularity, 0, 0));
  REQUIRE(ptrA != nullptr);

  hipMemGenericAllocationHandle_t handle = GetPhysicalMemory(device, size_mem);

  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));

  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));

  HIP_CHECK(hipMemcpy(ptrA, srcHostMem, size * sizeof(int), hipMemcpyHostToDevice));

  *reservedAddrSize = size_mem;
  return ptrA;
}

/**
 * Helper function to validate the handle from hipMemGetHandleForAddressRange
 * by extracting the data from the handle
 */
bool validateHandle(int handle, int size, int device = 0) {
  hipMemGenericAllocationHandle_t imported_handle;
  HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
            reinterpret_cast<void*>(static_cast<uintptr_t>(handle)),
            hipMemHandleTypePosixFileDescriptor));

  size_t granularity = GetGranularity(device);
  if (granularity <= 0) {
    std::cout << "Invalid Granularity" << std::endl;
    return false;
  }
  int sizeBytes = size * sizeof(int);
  size_t sizeMem = ((granularity + sizeBytes - 1) / granularity) * granularity;

  void* dstDevMem = nullptr;
  HIP_CHECK(hipMemAddressReserve(&dstDevMem, sizeMem, granularity, 0, 0));
  REQUIRE(dstDevMem != nullptr);
  HIP_CHECK(hipMemMap(dstDevMem, sizeMem, 0, imported_handle, 0));

  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(dstDevMem, sizeMem, &accessDesc, 1));

  int* dstHostMem = reinterpret_cast<int*>(malloc(sizeBytes));
  HIP_CHECK(hipMemcpy(dstHostMem, dstDevMem, sizeBytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < size; i++) {
    if (dstHostMem[i] != i) {
      std::cout << "Mismatch at " << i << " : " << dstHostMem[i] << std::endl;
      return false;
    }
  }

  hipLaunchKernelGGL(squareKernel, dim3(size / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                     static_cast<int*>(dstDevMem));
  HIP_CHECK(hipMemcpy(dstHostMem, dstDevMem, sizeBytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < size; i++) {
    if (dstHostMem[i] != (i * i)) {
      std::cout << "Mismatch at " << i << " : " << dstHostMem[i] << std::endl;
      return false;
    }
  }

  HIP_CHECK(hipMemUnmap(dstDevMem, sizeMem));
  HIP_CHECK(hipMemAddressFree(dstDevMem, sizeMem));
  free(dstHostMem);
  return true;
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario,
 *  - 1) Create the device memory
 *  - 2) Get handle from hipMemGetHandleForAddressRange
 *  - 3) Validate the handle by doing Read and Write operations
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_DeviceMemory") {
  constexpr int size = 1024;
  constexpr int sizeBytes = size * sizeof(int);

  void* srcDevMem = createDeviceMemoryAndFillData(size);
  REQUIRE(srcDevMem != nullptr);

  int handle = -1;
  HIP_CHECK(hipMemGetHandleForAddressRange(&handle, srcDevMem, sizeBytes,
                                           hipMemRangeHandleTypeDmaBufFd, 0));
  REQUIRE(handle > 0);

  hipDevice_t device;
  constexpr int kDeviceId = 0;
  HIP_CHECK(hipDeviceGet(&device, kDeviceId));
  checkVMMSupported(device);
  REQUIRE(validateHandle(handle, size) == true);

  HIP_CHECK(hipFree(srcDevMem));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario
 *  - 1) Create the Virtual memory
 *  - 2) Get handle from hipMemGetHandleForAddressRange
 *  - 3) Validate the handle by doing Read and Write operations
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_VM") {
  hipDevice_t device;
  constexpr int kDeviceId = 0;
  HIP_CHECK(hipDeviceGet(&device, kDeviceId));
  checkVMMSupported(device);

  constexpr int size = 1024;
  constexpr int sizeBytes = size * sizeof(int);

  hipDeviceptr_t ptrA = nullptr;
  int reservedAddrSize;
  ptrA = createVirtualMemoryAndFillData(size, &reservedAddrSize);
  REQUIRE(ptrA != nullptr);

  int handle = -1;
  HIP_CHECK(
      hipMemGetHandleForAddressRange(&handle, ptrA, sizeBytes, hipMemRangeHandleTypeDmaBufFd, 0));
  REQUIRE(handle > 0);

  REQUIRE(validateHandle(handle, size) == true);

  HIP_CHECK(hipMemUnmap(ptrA, reservedAddrSize));
  HIP_CHECK(hipMemAddressFree(ptrA, reservedAddrSize));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario
 *  - 1) Create the Device memory in a device
 *  - 2) Get handle from hipMemGetHandleForAddressRange in same device
 *  - 3) Using the handle do Read and Write operations in another device
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_DeviceMemory_InAnotherDevice") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  constexpr int srcDeviceId = 0;
  constexpr int dstDeviceId = 1;

  constexpr int size = 1024;
  constexpr int sizeBytes = size * sizeof(int);

  HIP_CHECK(hipSetDevice(srcDeviceId));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, srcDeviceId));
  checkVMMSupported(device);

  void* srcDevMem = nullptr;
  srcDevMem = createDeviceMemoryAndFillData(size);
  REQUIRE(srcDevMem != nullptr);

  int handle = -1;
  HIP_CHECK(hipMemGetHandleForAddressRange(&handle, srcDevMem, sizeBytes,
                                           hipMemRangeHandleTypeDmaBufFd, 0));
  REQUIRE(handle > 0);

  HIP_CHECK(hipSetDevice(dstDeviceId));

  HIP_CHECK(hipDeviceGet(&device, dstDeviceId));
  checkVMMSupported(device);
  REQUIRE(validateHandle(handle, size, dstDeviceId) == true);

  HIP_CHECK(hipFree(srcDevMem));
  HIP_CHECK(hipDeviceReset())
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario,
 *  - 1) Create the Virtual memory in a device
 *  - 2) Get handle from hipMemGetHandleForAddressRange in same device
 *  - 3) Using the handle do Read and Write operations in another device
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_VM_InAnotherDevice") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  constexpr int srcDeviceId = 0;
  constexpr int dstDeviceId = 1;

  HIP_CHECK(hipSetDevice(srcDeviceId));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, srcDeviceId));
  checkVMMSupported(device);

  constexpr int kNumElemsSize = 1024;
  constexpr int size = 1024;
  constexpr int sizeBytes = kNumElemsSize * sizeof(int);

  hipDeviceptr_t ptrA = nullptr;
  int reservedAddrSize;
  ptrA = createVirtualMemoryAndFillData(size, &reservedAddrSize);
  REQUIRE(ptrA != nullptr);

  int handle = 0;
  HIP_CHECK(
      hipMemGetHandleForAddressRange(&handle, ptrA, sizeBytes, hipMemRangeHandleTypeDmaBufFd, 0));
  REQUIRE(handle > 0);

  HIP_CHECK(hipSetDevice(dstDeviceId));

  HIP_CHECK(hipDeviceGet(&device, dstDeviceId));
  checkVMMSupported(device);

  REQUIRE(validateHandle(handle, size, dstDeviceId) == true);

  HIP_CHECK(hipMemUnmap(ptrA, reservedAddrSize));
  HIP_CHECK(hipMemAddressFree(ptrA, reservedAddrSize));

  HIP_CHECK(hipDeviceReset())
}

#if __linux__
/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario,
 *  - 1) Create the Device memory in Parent Process
 *  - 2) Get handle from hipMemGetHandleForAddressRange in Parent Process
 *  - 3) Share the handle to child process
 *  - 3) Do Read and Write operations Child process
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_MulProc_Socket_DeviceMem") {
  int fd[2], fdSig[2];
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdSig) == 0);

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);

    // Wait for parent process to create the socket.
    int size_mem = 0;
    REQUIRE(read(fd[0], &size_mem, sizeof(int)) >= 0);
    // Open Socket as client
    ipcSocketCom sockObj(false);
    int shHandle;
    // Signal Parent process that Child is ready to receive msg
    int sig = 0;
    REQUIRE(write(fdSig[1], &sig, sizeof(int)) >= 0);

    // receive message from parent provess
    checkSysCallErrors(sockObj.recvShareableHdl(&shHandle));
    hipMemGenericAllocationHandle_t imported_handle;
    // import the sareable handle
    HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
              reinterpret_cast<void*>(static_cast<uintptr_t>(shHandle)),
              hipMemHandleTypePosixFileDescriptor));

    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);

    // Validate the handle
    REQUIRE(validateHandle(shHandle, size_mem / sizeof(int)));

    checkSysCallErrors(sockObj.closeThisSock());
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);

    constexpr int size = 1024;
    constexpr int sizeBytes = size * sizeof(int);

    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);

    void* srcDevMem = nullptr;
    srcDevMem = createDeviceMemoryAndFillData(size);
    REQUIRE(srcDevMem != nullptr);

    int handle = -1;
    HIP_CHECK(hipMemGetHandleForAddressRange(&handle, srcDevMem, sizeBytes,
                                             hipMemRangeHandleTypeDmaBufFd, 0));

    int size_mem = sizeBytes;
    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    // Signal child process that socket is ready
    REQUIRE(write(fd[1], &size_mem, sizeof(size_t)) >= 0);
    // Wait for the child process to receive msg
    int sig = 0;
    REQUIRE(read(fdSig[0], &sig, sizeof(int)) >= 0);
    checkSysCallErrors(sockObj.sendShareableHdl(handle, pid));
    // Wait for child process to exit.
    int status;
    REQUIRE(wait(&status) >= 0);
    REQUIRE(status == 0);
    // Free all resources
    checkSysCallErrors(sockObj.closeThisSock());
    // HIP_CHECK(hipMemRelease(handle));
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario
 *  - 1) Create the Virtual memory in Parent Process
 *  - 2) Get handle from hipMemGetHandleForAddressRange in Parent Process
 *  - 3) Share the handle to child process
 *  - 3) Do Read and Write operations Child process
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_MulProc_Socket_VM") {
  int fd[2], fdSig[2];
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdSig) == 0);

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
    // Wait for parent process to create the socket.
    int size_mem = 0;
    REQUIRE(read(fd[0], &size_mem, sizeof(int)) >= 0);
    // Open Socket as client
    ipcSocketCom sockObj(false);
    int shHandle;
    // Signal Parent process that Child is ready to receive msg
    int sig = 0;
    REQUIRE(write(fdSig[1], &sig, sizeof(int)) >= 0);
    // receive message from parent provess
    checkSysCallErrors(sockObj.recvShareableHdl(&shHandle));
    hipMemGenericAllocationHandle_t imported_handle;
    // import the sareable handle
    HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
              reinterpret_cast<void*>(static_cast<uintptr_t>(shHandle)),
              hipMemHandleTypePosixFileDescriptor));
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);

    // Validate handle
    REQUIRE(validateHandle(shHandle, size_mem / sizeof(int)));

    checkSysCallErrors(sockObj.closeThisSock());
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);

    constexpr int N = 1024;
    int buffer_size = N * sizeof(int);

    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);

    hipDeviceptr_t ptrA = nullptr;
    int reservedAddrSize;
    ptrA = createVirtualMemoryAndFillData(N, &reservedAddrSize);
    REQUIRE(ptrA != nullptr);

    int handle = -1;
    HIP_CHECK(hipMemGetHandleForAddressRange(&handle, ptrA, buffer_size,
                                             hipMemRangeHandleTypeDmaBufFd, 0));
    REQUIRE(handle > 0);

    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    int size_mem = buffer_size;
    // Signal child process that socket is ready
    REQUIRE(write(fd[1], &size_mem, sizeof(int)) >= 0);
    // Wait for the child process to receive msg
    int sig = 0;
    REQUIRE(read(fdSig[0], &sig, sizeof(int)) >= 0);
    checkSysCallErrors(sockObj.sendShareableHdl(handle, pid));
    // Wait for child process to exit.
    int status;
    REQUIRE(wait(&status) >= 0);
    REQUIRE(status == 0);
    // Free all resources
    checkSysCallErrors(sockObj.closeThisSock());
    // HIP_CHECK(hipMemRelease(handle));
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);

    HIP_CHECK(hipMemUnmap(ptrA, reservedAddrSize));
    HIP_CHECK(hipMemAddressFree(ptrA, reservedAddrSize));
  }
}

#endif

/*
 * Helper function to create the Device Memory and get handle
 */
void launchForDevMem() {
  constexpr int size = 1024;
  constexpr int sizeBytes = size * sizeof(int);
  void* srcDevMem = createDeviceMemoryAndFillData(size);

  int handle = -1;
  HIP_CHECK(hipMemGetHandleForAddressRange(&handle, srcDevMem, sizeBytes,
                                           hipMemRangeHandleTypeDmaBufFd, 0));
  REQUIRE(handle > 0);
  HIP_CHECK(hipFree(srcDevMem));
}

/*
 * Helper function to create the Virtual Memory and get handle
 */
void launchForVM() {
  constexpr int size = 1024;
  constexpr int sizeBytes = size * sizeof(int);

  hipDeviceptr_t ptrA = nullptr;
  int reservedAddrSize;
  ptrA = createVirtualMemoryAndFillData(size, &reservedAddrSize);
  REQUIRE(ptrA != nullptr);

  int handle = -1;
  HIP_CHECK(
      hipMemGetHandleForAddressRange(&handle, ptrA, sizeBytes, hipMemRangeHandleTypeDmaBufFd, 0));
  REQUIRE(handle > 0);

  HIP_CHECK(hipMemUnmap(ptrA, reservedAddrSize));
  HIP_CHECK(hipMemAddressFree(ptrA, reservedAddrSize));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following behaviour of
 *  - hipMemGetHandleForAddressRange in the Multi threaded
 *  - scenario with both Device Memory and the Virtual Memory
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_MultipleThreads") {
  hipDevice_t device;
  constexpr int kDeviceId = 0;
  HIP_CHECK(hipDeviceGet(&device, kDeviceId));
  checkVMMSupported(device);

  const unsigned int threadsSupported = std::thread::hardware_concurrency();
  const int numberOfThreads = (threadsSupported >= 10) ? 10 : threadsSupported;

  std::vector<std::thread> threads;

  SECTION("For DeviceMemory") {
    for (int t = 0; t < numberOfThreads; t++) {
      threads.push_back(std::thread(launchForDevMem));
    }
  }

  SECTION("For VM") {
    for (int t = 0; t < numberOfThreads; t++) {
      threads.push_back(std::thread(launchForVM));
    }
  }

  for (int t = 0; (t < numberOfThreads) && (t < threads.size()); t++) {
    threads[t].join();
  }
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario
 *  - 1) Create the device memory of 5 integers
 *  - 2) Get handle from hipMemGetHandleForAddressRange for all offsets (0-4)
 *  - 3) Check the handle is valid or not
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_DifferentOffsets") {
  int handle;
  int size = 5;
  int sizeBytes = size * sizeof(int);
  int* dptr = nullptr;
  HIP_CHECK(hipMalloc(&dptr, sizeBytes));
  REQUIRE(dptr != nullptr);

  for (int i = 0; i < size; i++) {
    handle = -1;
    HIP_CHECK(hipMemGetHandleForAddressRange(&handle, dptr + i, sizeBytes - (i * sizeof(int)),
                                             hipMemRangeHandleTypeDmaBufFd, 0));
    REQUIRE(handle > 0);
  }

  HIP_CHECK(hipFree(dptr));
}
