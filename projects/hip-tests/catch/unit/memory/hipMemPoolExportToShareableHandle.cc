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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipMemPoolExportToShareableHandle hipMemPoolExportToShareableHandle
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemPoolExportToShareableHandle(void*                      shared_handle,
                                                 hipMemPool_t               mem_pool,
                                                 hipMemAllocationHandleType handle_type,
                                                 unsigned int               flags) ` -
 * Exports a memory pool to the requested handle type.
 */

#include "mempool_common.hh"

constexpr int DATA_SIZE = 1024 * 1024;
constexpr size_t byte_size = DATA_SIZE * sizeof(int);

/**
 Kernel to perform Square of input data.
 */
static __global__ void square_kernel(int* Buff) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int temp = Buff[i] * Buff[i];
  Buff[i] = temp;
}

/**
 Fill with input and expected output data.
 */
static void fill_data(std::vector<int>& A_h, std::vector<int>& B_h, std::vector<int>& C_h) {
  for (int i = 0; i < DATA_SIZE; i++) {
    A_h[i] = i % 1024;
    B_h[i] = 0;
    C_h[i] = A_h[i] * A_h[i];
  }
}

/**
 * Test Description
 * ------------------------
 *    - Create mempool handle and allocate a memory chunk. Export
 * the mempool and the pointer to the chunk. In the same process,
 * Import the handle and the pointer in the same process. Use the
 * pointer in kernel launch.
 * ------------------------
 *    - unit/memory/hipMemPoolExportImportToShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolExportToShareableHandle_SameProc") {
  hipMemPoolPtrExportData ptrExp;
  hipShareableHdl sharedHandle;
  std::vector<int> A_h(DATA_SIZE), B_h(DATA_SIZE), C_h(DATA_SIZE);
  fill_data(A_h, B_h, C_h);
  hipMemPoolProps pool_props{};
  hipMemPool_t mempool, mempoolImp;
  checkMempoolSupported(0) HIP_CHECK(hipSetDevice(0));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  // Create mempool
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.handleTypes = hipMemHandleTypePosixFileDescriptor;
  HIP_CHECK(hipMemPoolCreate(&mempool, &pool_props));
  // Allocate device memory from mempool
  int* A_d;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size, mempool, stream));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h.data(), byte_size, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  // Export mempool
  HIP_CHECK(hipMemPoolExportToShareableHandle(&sharedHandle, mempool,
                                              hipMemHandleTypePosixFileDescriptor, 0));
  // Export A_d
  HIP_CHECK(hipMemPoolExportPointer(&ptrExp, A_d));
  // Import mempool
  HIP_CHECK(hipMemPoolImportFromShareableHandle(&mempoolImp, (void*)sharedHandle,
                                                hipMemHandleTypePosixFileDescriptor, 0));
  // Import and use pointer
  void* ptrImp;
  HIP_CHECK(hipMemPoolImportPointer(&ptrImp, mempoolImp, &ptrExp));
  square_kernel<<<dim3(DATA_SIZE / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, stream>>>(
      (int*)ptrImp);
  HIP_CHECK(hipMemcpyAsync(B_h.data(), ptrImp, byte_size, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));
  HIP_CHECK(hipFree(ptrImp));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemPoolDestroy(mempool));
  HIP_CHECK(hipMemPoolDestroy(mempoolImp));
}

/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. Create mempool handle and
 * allocate a memory chunk. Export the mempool and the pointer to
 * the chunk. Import the mempool and the pointer in child process.
 * Copy data to the memory chunk and launch kernel to perform
 * operations on the data.
 * ------------------------
 *    - unit/memory/hipMemPoolExportImportToShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolExportToShareableHandle_ChldUseHdl") {
  std::vector<int> A_h(DATA_SIZE), B_h(DATA_SIZE), C_h(DATA_SIZE);
  fill_data(A_h, B_h, C_h);
  int fd[2], fdSig[2];
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdSig) == 0);

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
    // Wait for parent process to create the socket.
    hipMemPoolPtrExportData ptrExp;
    REQUIRE(read(fd[0], &ptrExp, sizeof(hipMemPoolPtrExportData)) >= 0);
    // Open Socket as client
    ipcSocketCom sockObj(false);
    // Signal Parent process that Child is ready to receive msg
    int sig = 0;
    REQUIRE(write(fdSig[1], &sig, sizeof(int)) >= 0);
    hipShareableHdl shdl;
    // receive message from parent provess
    checkSysCallErrors(sockObj.recvShareableHdl(&shdl));
    // Import mempool
    hipMemPool_t mempoolImp;
    HIP_CHECK(hipMemPoolImportFromShareableHandle(&mempoolImp, (void*)shdl,
                                                  hipMemHandleTypePosixFileDescriptor, 0));
    // Import and use pointer
    void* ptrImp;
    HIP_CHECK(hipMemPoolImportPointer(&ptrImp, mempoolImp, &ptrExp));
    square_kernel<<<dim3(DATA_SIZE / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0>>>(
        (int*)ptrImp);
    HIP_CHECK(hipStreamSynchronize(0));
    // Import and use pointer
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    checkSysCallErrors(sockObj.closeThisSock());
    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);

    hipMemPoolProps pool_props{};
    checkMempoolSupported(0)
        // Set property
        hipMemPool_t mempool;
    pool_props.allocType = hipMemAllocationTypePinned;
    pool_props.location.id = 0;
    pool_props.location.type = hipMemLocationTypeDevice;
    pool_props.handleTypes = hipMemHandleTypePosixFileDescriptor;
    HIP_CHECK(hipMemPoolCreate(&mempool, &pool_props));
    // Export mempool
    hipShareableHdl shdl;
    HIP_CHECK(
        hipMemPoolExportToShareableHandle(&shdl, mempool, hipMemHandleTypePosixFileDescriptor, 0));
    // Allocate device memory from mempool
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    int* A_d;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size, mempool, stream));
    HIP_CHECK(hipMemcpyAsync(A_d, A_h.data(), byte_size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    hipMemPoolPtrExportData ptrExp;
    // Export A_d
    HIP_CHECK(hipMemPoolExportPointer(&ptrExp, A_d));
    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    // Signal child process that socket is ready and share ptr to child
    REQUIRE(write(fd[1], &ptrExp, sizeof(hipMemPoolPtrExportData)) >= 0);
    // Wait for the child process to receive msg
    int sig = 0;
    REQUIRE(read(fdSig[0], &sig, sizeof(int)) >= 0);
    checkSysCallErrors(sockObj.sendShareableHdl(shdl, pid));
    // Wait for child process to exit.
    int status;
    REQUIRE(wait(&status) >= 0);
    REQUIRE(status == 0);
    HIP_CHECK(hipMemcpyAsync(B_h.data(), A_d, byte_size, hipMemcpyDeviceToHost, stream));
    // Free all resources
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipMemPoolDestroy(mempool));
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
    checkSysCallErrors(sockObj.closeThisSock());
  }
}

/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. Create mempool handle and
 * allocate a memory chunk. Export the mempool and the pointer to
 * the chunk. Import the mempool and the pointer in child process.
 * In parent process change mempool property. Verify the change in
 * child process.
 * ------------------------
 *    - unit/memory/hipMemPoolExportImportToShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
#if HT_AMD
TEST_CASE("Unit_hipMemPoolExportToShareableHandle_ChldCheckAccess") {
  int fd[2], fdSig[2];
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdSig) == 0);

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
    // Wait for parent process to create the socket.
    int sig = 0;
    REQUIRE(read(fd[0], &sig, sizeof(int)) >= 0);
    // Open Socket as client
    ipcSocketCom sockObj(false);
    // Signal Parent process that Child is ready to receive msg
    REQUIRE(write(fdSig[1], &sig, sizeof(int)) >= 0);
    hipShareableHdl shdl;
    // receive message from parent provess
    checkSysCallErrors(sockObj.recvShareableHdl(&shdl));
    // Import mempool
    hipMemPool_t mempoolImp;
    HIP_CHECK(hipMemPoolImportFromShareableHandle(&mempoolImp, (void*)shdl,
                                                  hipMemHandleTypePosixFileDescriptor, 0));
    // Get and validate access for all devices
    int numDevices = 0;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    for (int dev = 0; dev < numDevices; dev++) {
      hipMemAccessFlags flags;
      hipMemLocation location;
      location.type = hipMemLocationTypeDevice;
      location.id = dev;
      HIP_CHECK(hipMemPoolGetAccess(&flags, mempoolImp, &location));
      REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
    }
    // Import and use pointer
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    checkSysCallErrors(sockObj.closeThisSock());
    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);

    hipMemPoolProps pool_props{};
    checkMempoolSupported(0)
        // Set property
        hipMemPool_t mempool;
    pool_props.allocType = hipMemAllocationTypePinned;
    pool_props.location.id = 0;
    pool_props.location.type = hipMemLocationTypeDevice;
    pool_props.handleTypes = hipMemHandleTypePosixFileDescriptor;
    HIP_CHECK(hipMemPoolCreate(&mempool, &pool_props));
    // Set access to all devices
    int numDevices = 0;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    for (int dev = 0; dev < numDevices; dev++) {
      checkMempoolSupported(dev) hipMemAccessDesc accessDesc;
      accessDesc.location.type = hipMemLocationTypeDevice;
      accessDesc.location.id = dev;
      accessDesc.flags = hipMemAccessFlagsProtReadWrite;
      HIP_CHECK(hipMemPoolSetAccess(mempool, &accessDesc, 1));
    }
    // Export mempool
    hipShareableHdl shdl;
    HIP_CHECK(
        hipMemPoolExportToShareableHandle(&shdl, mempool, hipMemHandleTypePosixFileDescriptor, 0));
    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    // Signal child process that socket is ready
    int sig = 0;
    REQUIRE(write(fd[1], &sig, sizeof(int)) >= 0);
    // Wait for the child process to receive msg
    REQUIRE(read(fdSig[0], &sig, sizeof(int)) >= 0);
    checkSysCallErrors(sockObj.sendShareableHdl(shdl, pid));
    // Wait for child process to exit.
    int status;
    REQUIRE(wait(&status) >= 0);
    REQUIRE(status == 0);
    HIP_CHECK(hipMemPoolDestroy(mempool));
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
    checkSysCallErrors(sockObj.closeThisSock());
  }
}
#endif
/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. Create mempool handle and
 * allocate a memory chunk. Export the mempool and the pointer to
 * the chunk. Import the mempool and the pointer in grandchild process.
 * Copy data to the memory chunk and launch kernel to perform
 * operations on the data.

 * ------------------------
 *    - unit/memory/hipMemPoolExportImportToShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolExportToShareableHandle_GrndChldUseHdl") {
  std::vector<int> A_h(DATA_SIZE), B_h(DATA_SIZE), C_h(DATA_SIZE);
  fill_data(A_h, B_h, C_h);
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
      hipMemPoolPtrExportData ptrExp;
      REQUIRE(read(fd[0], &ptrExp, sizeof(hipMemPoolPtrExportData)) >= 0);
      // Open Socket as client
      ipcSocketCom sockObj(false);
      hipShareableHdl shdl;
      // Signal Parent process that Child is ready to receive msg
      int sig = 0;
      REQUIRE(write(fdSig[1], &sig, sizeof(int)) >= 0);
      // receive message from parent provess
      checkSysCallErrors(sockObj.recvShareableHdl(&shdl));
      // Import mempool
      hipMemPool_t mempoolImp;
      HIP_CHECK(hipMemPoolImportFromShareableHandle(&mempoolImp, (void*)shdl,
                                                    hipMemHandleTypePosixFileDescriptor, 0));
      // Import and use pointer
      void* ptrImp;
      HIP_CHECK(hipMemPoolImportPointer(&ptrImp, mempoolImp, &ptrExp));
      square_kernel<<<dim3(DATA_SIZE / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0>>>(
          (int*)ptrImp);
      HIP_CHECK(hipStreamSynchronize(0));
      REQUIRE(close(fd[0]) == 0);
      REQUIRE(close(fdSig[1]) == 0);
      checkSysCallErrors(sockObj.closeThisSock());
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

    hipMemPoolProps pool_props{};
    checkMempoolSupported(0)
        // Set property
        hipMemPool_t mempool;
    pool_props.allocType = hipMemAllocationTypePinned;
    pool_props.location.id = 0;
    pool_props.location.type = hipMemLocationTypeDevice;
    pool_props.handleTypes = hipMemHandleTypePosixFileDescriptor;
    HIP_CHECK(hipMemPoolCreate(&mempool, &pool_props));
    // Export mempool
    hipShareableHdl shdl;
    HIP_CHECK(
        hipMemPoolExportToShareableHandle(&shdl, mempool, hipMemHandleTypePosixFileDescriptor, 0));
    // Allocate device memory from mempool
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    int* A_d;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size, mempool, stream));
    HIP_CHECK(hipMemcpyAsync(A_d, A_h.data(), byte_size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    hipMemPoolPtrExportData ptrExp;
    // Export A_d
    HIP_CHECK(hipMemPoolExportPointer(&ptrExp, A_d));

    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    // Signal child process that socket is ready and share ptr to child
    REQUIRE(write(fd[1], &ptrExp, sizeof(hipMemPoolPtrExportData)) >= 0);
    // Wait for the child process to receive msg
    int sig = 0;
    REQUIRE(read(fdSig[0], &sig, sizeof(int)) >= 0);
    checkSysCallErrors(sockObj.sendShareableHdl(shdl, pid_grChld));
    // Wait for child process to exit.
    int status;
    REQUIRE(wait(&status) >= 0);
    REQUIRE(status == 0);
    HIP_CHECK(hipMemcpyAsync(B_h.data(), A_d, byte_size, hipMemcpyDeviceToHost, stream));
    // Free all resources
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));
    // Free all resources
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipMemPoolDestroy(mempool));
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
    REQUIRE(close(fdpid[0]) == 0);
    checkSysCallErrors(sockObj.closeThisSock());
  }
}
/**
 * Test Description
 * ------------------------
 *    - Negative Tests for hipMemPoolExportToShareableHandle.
 * ------------------------
 *    - unit/memory/hipMemPoolExportImportToShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolExportToShareableHandle_Negative") {
  hipShareableHdl sharedHandle;
  hipMemPoolProps pool_props{};
  hipMemPool_t mempoolPfd, mempoolwoPfd;
  checkMempoolSupported(0)

      // Create mempool with Posix File Descriptor
      pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.handleTypes = hipMemHandleTypePosixFileDescriptor;
  HIP_CHECK(hipMemPoolCreate(&mempoolPfd, &pool_props));

  // Create mempool without File Descriptor
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.handleTypes = hipMemHandleTypeNone;
  HIP_CHECK(hipMemPoolCreate(&mempoolwoPfd, &pool_props));
  SECTION("Passing nullptr as handle") {
    HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(nullptr, mempoolPfd,
                                                      hipMemHandleTypePosixFileDescriptor, 0),
                    hipErrorInvalidValue);
  }
  SECTION("Passing nullptr as mempool") {
    HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(&sharedHandle, nullptr,
                                                      hipMemHandleTypePosixFileDescriptor, 0),
                    hipErrorInvalidValue);
  }
  SECTION("Passing invalid handle type") {
    HIP_CHECK_ERROR(
        hipMemPoolExportToShareableHandle(&sharedHandle, mempoolPfd, hipMemHandleTypeNone, 0),
        hipErrorInvalidValue);
  }
  SECTION("Passing mempool without file descriptor") {
    HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(&sharedHandle, mempoolwoPfd,
                                                      hipMemHandleTypePosixFileDescriptor, 0),
                    hipErrorInvalidValue);
  }
  HIP_CHECK(hipMemPoolDestroy(mempoolPfd));
  HIP_CHECK(hipMemPoolDestroy(mempoolwoPfd));
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
