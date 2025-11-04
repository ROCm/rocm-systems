/*
Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipMemExportToShareableHandle hipMemExportToShareableHandle
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t *handle,
 *                                             void *osHandle,
 *                                             hipMemAllocationHandleType shHandleType)` -
 * 	Imports an allocation from a requested shareable handle type.
 */

#include <hip_test_common.hh>
#include "hip_vmm_common.hh"

#define DATA_SIZE (1 << 13)
#define THREADS_PER_BLOCK 512
typedef int ShareableHandle;

/**
 Kernel to perform Square of input data.
 */
static __global__ void square_kernel(int* Buff) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int temp = Buff[i] * Buff[i];
  Buff[i] = temp;
}

/**
 * Test Description
 * ------------------------
 *    - Basic sanity test.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemImportFromShareableHandle_Positive_Basic") {
  CTX_DESTROY();

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t granularity;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, granularity * 2, &prop, 0));
  ShareableHandle shareable_handle;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));
  hipMemGenericAllocationHandle_t imported_handle;
  HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
            reinterpret_cast<void*>(static_cast<uintptr_t>(shareable_handle)),
            hipMemHandleTypePosixFileDescriptor));
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemRelease(imported_handle));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemImportFromShareableHandle_Negative_Parameters") {
  CTX_CREATE();

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t granularity;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, granularity * 2, &prop, 0));

  void* shareable_handle = nullptr;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));

  hipMemGenericAllocationHandle_t imported_handle;

  SECTION("handle == nullptr") {
    HIP_CHECK_ERROR(hipMemImportFromShareableHandle(nullptr, shareable_handle,
                                                    hipMemHandleTypePosixFileDescriptor),
                    hipErrorInvalidValue);
  }

  SECTION("shareableHandle == nullptr") {
    HIP_CHECK_ERROR(hipMemImportFromShareableHandle(&imported_handle, nullptr,
                                                    hipMemHandleTypePosixFileDescriptor),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. Create Vmm handle in Parent
 * Process and export it to Child Process using Sockets. The Child
 * Process imports this handle via sockets and uses this handle
 * to perform VMM operations.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemImportFromShareableHandle_MulProc_ChldUseHdl") {
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int fd[2], fdSig[2];
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdSig) == 0);

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
    // Wait for parent process to create the socket.
    size_t size_mem = 0;
    REQUIRE(read(fd[0], &size_mem, sizeof(size_t)) >= 0);
    // Open Socket as client
    ipcSocketCom sockObj(false);
    hipShareableHdl shHandle;
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
    // Allocate virtual address range
    hipDeviceptr_t ptrA;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, imported_handle, 0));
    // Set access
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = 0;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    std::vector<int> A_h(N), B_h(N), C_h(N);
    // Initialize with data
    for (size_t idx = 0; idx < N; idx++) {
      A_h[idx] = idx;
      C_h[idx] = idx * idx;
    }
    HIP_CHECK(hipMemcpyHtoD(ptrA, A_h.data(), buffer_size));
    // Invoke kernel
    hipLaunchKernelGGL(square_kernel, dim3(N / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                       reinterpret_cast<int*>(ptrA));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA, buffer_size));
    HIP_CHECK(hipDeviceSynchronize());
    // validate
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));

    // free resources
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
    checkSysCallErrors(sockObj.closeThisSock());
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);

    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);
    // Set property
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    // Set Granularity of the VMM memory
    size_t granularity;
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle;
    HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

    hipShareableHdl shareable_handle;
    HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                            hipMemHandleTypePosixFileDescriptor, 0));
    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    // Signal child process that socket is ready
    REQUIRE(write(fd[1], &size_mem, sizeof(size_t)) >= 0);
    // Wait for the child process to receive msg
    int sig = 0;
    REQUIRE(read(fdSig[0], &sig, sizeof(int)) >= 0);
    checkSysCallErrors(sockObj.sendShareableHdl(shareable_handle, pid));
    // Wait for child process to exit.
    int status;
    REQUIRE(wait(&status) >= 0);
    REQUIRE(status == 0);
    // Free all resources
    checkSysCallErrors(sockObj.closeThisSock());
    HIP_CHECK(hipMemRelease(handle));
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. Create Vmm handle in Parent
 * Process and export it to Child Process using Sockets. The Child
 * Process imports this handle via sockets. Both Parent and Child Process
 * uses this handle to perform VMM operations.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemImportFromShareableHandle_MulProc_ParntChldUseHdl") {
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int fd[2], fdSig[2];
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdSig) == 0);

  // Create data buffer
  std::vector<int> A_h(N), B_h(N), C_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
    C_h[idx] = idx * idx;
  }

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
    // Wait for parent process to create the socket.
    size_t size_mem = 0;
    REQUIRE(read(fd[0], &size_mem, sizeof(size_t)) >= 0);

    // Open Socket as client
    ipcSocketCom sockObj(false);
    hipShareableHdl shHandle;

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
    // Allocate virtual address range
    hipDeviceptr_t ptrA;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, imported_handle, 0));
    // Set access
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = 0;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    HIP_CHECK(hipMemcpyHtoD(ptrA, A_h.data(), buffer_size));
    // Invoke kernel
    hipLaunchKernelGGL(square_kernel, dim3(N / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                       reinterpret_cast<int*>(ptrA));
    HIP_CHECK(hipDeviceSynchronize());

    // free resources
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
    checkSysCallErrors(sockObj.closeThisSock());
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);
    // Set property
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    // Set Granularity of the VMM memory
    size_t granularity;
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle;
    HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

    hipShareableHdl shareable_handle;
    HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                            hipMemHandleTypePosixFileDescriptor, 0));

    // Allocate virtual address range
    hipDeviceptr_t ptrA;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
    // Set access
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = device;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));

    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    // Signal child process that socket is ready
    REQUIRE(write(fd[1], &size_mem, sizeof(size_t)) >= 0);
    // Wait for the child process to receive msg
    int sig = 0;
    REQUIRE(read(fdSig[0], &sig, sizeof(int)) >= 0);
    checkSysCallErrors(sockObj.sendShareableHdl(shareable_handle, pid));
    // Wait for child process to exit.
    int status;
    REQUIRE(wait(&status) >= 0);
    REQUIRE(status == 0);

    // Check results of Vmm data processing in child
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA, buffer_size));
    // validate
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));

    // Free all resources
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
    HIP_CHECK(hipMemRelease(handle));
    checkSysCallErrors(sockObj.closeThisSock());
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. Create Vmm handle in Parent
 * Process and export it to Grand Child Process using Sockets. The Grand
 * Child Process imports this handle via sockets. The Grand Child Process
 * uses this handle to perform VMM operations.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemImportFromShareableHandle_MulProc_GrndChldUseHdl") {
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int fd[2], fdSig[2], fdpid[2];
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdSig) == 0);
  REQUIRE(pipe(fdpid) == 0);
  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    auto pid2 = fork();
    if (pid2 == 0) {  // grandchild
      REQUIRE(close(fd[1]) == 0);
      REQUIRE(close(fdSig[0]) == 0);
      // Wait for parent process to create the socket.
      size_t size_mem = 0;
      REQUIRE(read(fd[0], &size_mem, sizeof(size_t)) >= 0);

      // Open Socket as client
      ipcSocketCom sockObj(false);
      hipShareableHdl shHandle;

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
      // Allocate virtual address range
      hipDeviceptr_t ptrA;
      HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
      HIP_CHECK(hipMemMap(ptrA, size_mem, 0, imported_handle, 0));
      // Set access
      hipMemAccessDesc accessDesc = {};
      accessDesc.location.type = hipMemLocationTypeDevice;
      accessDesc.location.id = 0;
      accessDesc.flags = hipMemAccessFlagsProtReadWrite;
      // Make the address accessible to GPU 0
      HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
      std::vector<int> A_h(N), B_h(N), C_h(N);
      // Initialize with data
      for (size_t idx = 0; idx < N; idx++) {
        A_h[idx] = idx;
        C_h[idx] = idx * idx;
      }
      HIP_CHECK(hipMemcpyHtoD(ptrA, A_h.data(), buffer_size));
      // Invoke kernel
      hipLaunchKernelGGL(square_kernel, dim3(N / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                         reinterpret_cast<int*>(ptrA));
      HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA, buffer_size));
      HIP_CHECK(hipDeviceSynchronize());
      // validate
      REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));

      // free resources
      HIP_CHECK(hipMemUnmap(ptrA, size_mem));
      HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
      checkSysCallErrors(sockObj.closeThisSock());
      REQUIRE(close(fd[0]) == 0);
      REQUIRE(close(fdSig[1]) == 0);
      exit(0);
    } else {
      int status;
      REQUIRE(close(fdpid[0]) == 0);
      REQUIRE(write(fdpid[1], &pid2, sizeof(pid2)) >= 0);
      REQUIRE(wait(&status) >= 0);
      REQUIRE(status == 0);
      REQUIRE(close(fdpid[1]) == 0);
      exit(0);
    }
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    REQUIRE(close(fdpid[1]) == 0);
    int pid_grChld = 0;
    REQUIRE(read(fdpid[0], &pid_grChld, sizeof(pid_grChld)) >= 0);

    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);
    // Set property
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    // Set Granularity of the VMM memory
    size_t granularity;
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle;
    HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

    hipShareableHdl shareable_handle;
    HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                            hipMemHandleTypePosixFileDescriptor, 0));

    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    // Signal child process that socket is ready
    REQUIRE(write(fd[1], &size_mem, sizeof(size_t)) >= 0);
    // Wait for the child process to receive msg
    int sig = 0;
    REQUIRE(read(fdSig[0], &sig, sizeof(int)) >= 0);
    checkSysCallErrors(sockObj.sendShareableHdl(shareable_handle, pid_grChld));
    // Wait for child process to exit.
    int status;
    REQUIRE(wait(&status) >= 0);
    REQUIRE(status == 0);

    // Free all resources
    HIP_CHECK(hipMemRelease(handle));
    checkSysCallErrors(sockObj.closeThisSock());
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
    REQUIRE(close(fdpid[0]) == 0);
  }
}

TEST_CASE("Unit_hipMemImportFromShareableHandle_Capture") {
  CTX_CREATE();

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp allocation_prop = {};
  allocation_prop.type = hipMemAllocationTypePinned;
  allocation_prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  allocation_prop.location.type = hipMemLocationTypeDevice;
  allocation_prop.location.id = device;

  size_t granularity = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &allocation_prop,
                                           hipMemAllocationGranularityMinimum));

  hipMemGenericAllocationHandle_t allocation_handle;
  HIP_CHECK(hipMemCreate(&allocation_handle, granularity * 2, &allocation_prop, 0));
  ShareableHandle shareable_handle;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, allocation_handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));
  hipMemGenericAllocationHandle_t imported_handle;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
            reinterpret_cast<void*>(static_cast<uintptr_t>(shareable_handle)),
            hipMemHandleTypePosixFileDescriptor));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemRelease(allocation_handle));
  CTX_DESTROY();
}
/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
