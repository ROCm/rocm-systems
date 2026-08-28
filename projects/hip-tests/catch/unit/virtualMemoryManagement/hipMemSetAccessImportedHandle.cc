/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemSetAccess hipMemSetAccess
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc,
 *                             size_t count)` -
 * 	Sets the access flags for each location specified in desc for the given virtual
 * address range.
 *
 * These tests cover granting *CPU* access (hipMemLocationTypeHost) to a mapping backed by
 * an allocation that was created in another process and brought in with
 * hipMemImportFromShareableHandle. An imported allocation carries no CPU mapping of its
 * own, so this exercises a different path than the same call on a locally created handle
 * (covered by Unit_hipMemSetAccessHost_devicealloc).
 */

#include <hip_test_common.hh>
#include "hip_vmm_common.hh"

#define DATA_SIZE (1 << 13)

// Device access on the imported mapping is exercised with hipMemsetD32 rather than a
// kernel launch: these tests run their HIP work in a fork()ed child, and loading a code
// object there is unreliable.
#define MEMSET_PATTERN 0xdeadbeef

/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. The Parent Process creates a device backed Vmm
 * allocation, seeds it and exports it to the Child Process over a socket. The Child
 * Process imports the handle, maps it into its own reserved address range and grants
 * *CPU* access to the imported mapping. The Child then reads the Parent's data and writes
 * back through the CPU mapping, and the Parent verifies the result.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetAccessImportedHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemSetAccess_MulProc_ImportedDeviceMem_HostAccess) {
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
    CTX_CREATE();
    // Wait for parent process to create the socket.
    size_t size_mem = 0;
    REQUIRE(read(fd[0], &size_mem, sizeof(size_t)) >= 0);
    // Open Socket as client
    ipcSocketCom sockObj(false);
    hipShareableHdl shHandle;
    // Signal Parent process that Child is ready to receive msg
    int sig = 0;
    REQUIRE(write(fdSig[1], &sig, sizeof(int)) >= 0);
    // receive message from parent process
    checkSysCallErrors(sockObj.recvShareableHdl(&shHandle));
    hipMemGenericAllocationHandle_t imported_handle;
    // import the shareable handle
    HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
              reinterpret_cast<void*>(static_cast<uintptr_t>(shHandle)),
              hipMemHandleTypePosixFileDescriptor));
    // Allocate virtual address range and map the imported allocation into it
    void* ptrA;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, imported_handle, 0));

    // Grant CPU access to the imported mapping. This is the case under test: the imported
    // allocation has no CPU mapping of its own, so one has to be established here.
    hipMemAccessDesc accHost = {};
    accHost.location.type = hipMemLocationTypeHost;
    accHost.location.id = 0;
    accHost.flags = hipMemAccessFlagsProtReadWrite;
#if HT_AMD
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accHost, 1));

    // The CPU can now read what the parent seeded and write back through the mapping.
    int* hostPtr = reinterpret_cast<int*>(ptrA);
    std::vector<int> expected(N);
    for (size_t idx = 0; idx < N; idx++) expected[idx] = idx;
    REQUIRE(true == std::equal(expected.begin(), expected.end(), hostPtr));

    for (size_t idx = 0; idx < N; idx++) hostPtr[idx] = static_cast<int>(idx) * 2;
#else
    // CUDA does not allow host access to a device located allocation.
    HIP_CHECK_ERROR(hipMemSetAccess(ptrA, size_mem, &accHost, 1), hipErrorNotSupported);
#endif

    // free resources
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
    HIP_CHECK(hipMemRelease(imported_handle));
    CTX_DESTROY();
    checkSysCallErrors(sockObj.closeThisSock());
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    CTX_CREATE();

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

    // Map it locally and seed it so the child has something to read back.
    void* ptrA;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = device;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));

    std::vector<int> A_h(N), B_h(N);
    for (size_t idx = 0; idx < N; idx++) A_h[idx] = idx;
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));

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

#if HT_AMD
    // Check what the child wrote through its CPU mapping of the imported allocation.
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    std::vector<int> expected(N);
    for (size_t idx = 0; idx < N; idx++) expected[idx] = static_cast<int>(idx) * 2;
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), expected.data()));
#endif

    // Free all resources
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
    HIP_CHECK(hipMemRelease(handle));
    checkSysCallErrors(sockObj.closeThisSock());
    CTX_DESTROY();
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. Same import as above, but the Child Process grants
 * CPU access to the imported mapping *before* granting GPU access, seeds it through the CPU
 * mapping and then overwrites it from the device with hipMemsetD32. This checks that the host
 * and device mappings of an imported allocation coexist and that granting host access first
 * does not disturb the subsequent device mapping.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetAccessImportedHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemSetAccess_MulProc_ImportedDeviceMem_HostThenDeviceAccess) {
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
    CTX_CREATE();
    // Wait for parent process to create the socket.
    size_t size_mem = 0;
    REQUIRE(read(fd[0], &size_mem, sizeof(size_t)) >= 0);
    // Open Socket as client
    ipcSocketCom sockObj(false);
    hipShareableHdl shHandle;
    // Signal Parent process that Child is ready to receive msg
    int sig = 0;
    REQUIRE(write(fdSig[1], &sig, sizeof(int)) >= 0);
    // receive message from parent process
    checkSysCallErrors(sockObj.recvShareableHdl(&shHandle));
    hipMemGenericAllocationHandle_t imported_handle;
    // import the shareable handle
    HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
              reinterpret_cast<void*>(static_cast<uintptr_t>(shHandle)),
              hipMemHandleTypePosixFileDescriptor));
    // Allocate virtual address range and map the imported allocation into it
    void* ptrA;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, imported_handle, 0));

    // Grant CPU access first, then GPU access, on the same imported mapping.
    hipMemAccessDesc accHost = {};
    accHost.location.type = hipMemLocationTypeHost;
    accHost.location.id = 0;
    accHost.flags = hipMemAccessFlagsProtReadWrite;
#if HT_AMD
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accHost, 1));

    hipMemAccessDesc accDev = {};
    accDev.location.type = hipMemLocationTypeDevice;
    accDev.location.id = 0;
    accDev.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accDev, 1));

    // Seed through the CPU mapping, overwrite from the GPU, then read back through the CPU
    // mapping. Both mappings of the imported allocation have to be live for this to hold.
    int* hostPtr = reinterpret_cast<int*>(ptrA);
    for (size_t idx = 0; idx < N; idx++) hostPtr[idx] = static_cast<int>(idx);

    HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(ptrA), MEMSET_PATTERN, N));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int> expected(N, static_cast<int>(MEMSET_PATTERN));
    REQUIRE(true == std::equal(expected.begin(), expected.end(), hostPtr));

    // And the device side observes the same memory through hipMemcpyDtoH.
    std::vector<int> readback(N, 0);
    HIP_CHECK(hipMemcpyDtoH(readback.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    REQUIRE(true == std::equal(readback.begin(), readback.end(), expected.data()));
#else
    // CUDA does not allow host access to a device located allocation.
    HIP_CHECK_ERROR(hipMemSetAccess(ptrA, size_mem, &accHost, 1), hipErrorNotSupported);
#endif

    // free resources
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
    HIP_CHECK(hipMemRelease(imported_handle));
    CTX_DESTROY();
    checkSysCallErrors(sockObj.closeThisSock());
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    CTX_CREATE();

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
    HIP_CHECK(hipMemRelease(handle));
    checkSysCallErrors(sockObj.closeThisSock());
    CTX_DESTROY();
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
  }
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
