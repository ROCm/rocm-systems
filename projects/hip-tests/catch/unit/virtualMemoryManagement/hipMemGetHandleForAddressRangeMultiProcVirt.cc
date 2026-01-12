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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <utils.hh>

#if __linux__
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "hip_vmm_common.hh"
#include "hipMemGetHandleForAddressRange_common.hh"



#if __linux__

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
    CTX_CREATE();
    // Wait for parent process to create the socket.
    int size_mem = 0;
    REQUIRE(read(fd[0], &size_mem, sizeof(int)) >= 0);
    // Open Socket as client
    ipcSocketCom sockObj(false);
    int shHandle;
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
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);

    // Validate handle
    REQUIRE(validateHandle(shHandle, size_mem / sizeof(int)));
    CTX_DESTROY();

    checkSysCallErrors(sockObj.closeThisSock());
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    CTX_CREATE();

    constexpr int N = 1024;
    int buffer_size = N * sizeof(int);

    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);

    hipDeviceptr_t ptrA;
    int reservedAddrSize;
    ptrA = createVirtualMemoryAndFillData(N, &reservedAddrSize);
    REQUIRE(reinterpret_cast<void*>(ptrA) != nullptr);

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

    HIP_CHECK(hipMemUnmap(reinterpret_cast<void*>(ptrA), reservedAddrSize));
    HIP_CHECK(hipMemAddressFree(reinterpret_cast<void*>(ptrA), reservedAddrSize));
    CTX_DESTROY();
  }
}

#endif


