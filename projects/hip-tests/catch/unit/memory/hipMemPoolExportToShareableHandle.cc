/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
#include <hip_test_process.hh>

static int DATA_SIZE() {
  static const int val = isQuickLevel() ? 128 * 1024 : 1024 * 1024;
  return val;
}
static size_t byte_size() { return DATA_SIZE() * sizeof(int); }

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
  for (int i = 0; i < DATA_SIZE(); i++) {
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
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolExportToShareableHandle_SameProc) {
  hipMemPoolPtrExportData ptrExp;
  hipShareableHdl sharedHandle;
  std::vector<int> A_h(DATA_SIZE()), B_h(DATA_SIZE()), C_h(DATA_SIZE());
  fill_data(A_h, B_h, C_h);
  hipMemPoolProps pool_props{};
  hipMemPool_t mempool, mempoolImp;
  checkMempoolSupported(0) HIP_CHECK(hipSetDevice(0));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  #if HT_WIN
  hipMemAllocationHandleType handleType = hipMemHandleTypeWin32;
  #else
  hipMemAllocationHandleType handleType = hipMemHandleTypePosixFileDescriptor;
  #endif
  // Create mempool
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.handleTypes = handleType;
  HIP_CHECK(hipMemPoolCreate(&mempool, &pool_props));
  // Allocate device memory from mempool
  int* A_d;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size(), mempool, stream));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h.data(), byte_size(), hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  // Export mempool
  HIP_CHECK(hipMemPoolExportToShareableHandle(&sharedHandle, mempool,
                                              handleType, 0));
  // Export A_d
  HIP_CHECK(hipMemPoolExportPointer(&ptrExp, A_d));
  // Import mempool
  HIP_CHECK(hipMemPoolImportFromShareableHandle(&mempoolImp, (void*)sharedHandle,
                                                handleType, 0));
  // Import and use pointer
  void* ptrImp;
  HIP_CHECK(hipMemPoolImportPointer(&ptrImp, mempoolImp, &ptrExp));
  square_kernel<<<dim3(DATA_SIZE() / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, stream>>>(
      (int*)ptrImp);
  HIP_CHECK(hipMemcpyAsync(B_h.data(), ptrImp, byte_size(), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));
  HIP_CHECK(hipFree(ptrImp));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemPoolDestroy(mempool));
  HIP_CHECK(hipMemPoolDestroy(mempoolImp));
}

#if HT_LINUX
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
HIP_TEST_CASE(Unit_hipMemPoolExportToShareableHandle_ChldUseHdl) {
  std::vector<int> A_h(DATA_SIZE()), B_h(DATA_SIZE()), C_h(DATA_SIZE());
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
    square_kernel<<<dim3(DATA_SIZE() / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0>>>(
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
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size(), mempool, stream));
    HIP_CHECK(hipMemcpyAsync(A_d, A_h.data(), byte_size(), hipMemcpyHostToDevice, stream));
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
    HIP_CHECK(hipMemcpyAsync(B_h.data(), A_d, byte_size(), hipMemcpyDeviceToHost, stream));
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
HIP_TEST_CASE(Unit_hipMemPoolExportToShareableHandle_ChldCheckAccess) {
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
HIP_TEST_CASE(Unit_hipMemPoolExportToShareableHandle_GrndChldUseHdl) {
  std::vector<int> A_h(DATA_SIZE()), B_h(DATA_SIZE()), C_h(DATA_SIZE());
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
      square_kernel<<<dim3(DATA_SIZE() / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0>>>(
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
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size(), mempool, stream));
    HIP_CHECK(hipMemcpyAsync(A_d, A_h.data(), byte_size(), hipMemcpyHostToDevice, stream));
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
    HIP_CHECK(hipMemcpyAsync(B_h.data(), A_d, byte_size(), hipMemcpyDeviceToHost, stream));
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
#endif  // HT_LINUX

/**
 * Test Description
 * ------------------------
 *    - Cross-platform multiprocess test. Parent creates a mempool with IPC-capable
 * handle type, allocates memory, fills it with data, and exports the handle and
 * pointer via shared memory. A child process imports the handle and pointer,
 * runs a square kernel, and exits. The parent verifies the results.
 * ------------------------
 *    - unit/memory/hipMemPoolExportToShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipMemPoolExportToShareableHandle_multiproc) {
  checkMempoolSupported(0)
  HIP_CHECK(hipSetDevice(0));

  int handleTypesSupported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&handleTypesSupported,
      hipDeviceAttributeMemoryPoolSupportedHandleTypes, 0));

  hipMemAllocationHandleType handleType;
#if HT_WIN
  if (!(handleTypesSupported & hipMemHandleTypeWin32)) {
    HipTest::HIP_SKIP_TEST("Win32 handle type not supported. Skipping Test..");
    return;
  }
  handleType = hipMemHandleTypeWin32;
#else
  if (!(handleTypesSupported & hipMemHandleTypePosixFileDescriptor)) {
    HipTest::HIP_SKIP_TEST("POSIX FD handle type not supported. Skipping Test..");
    return;
  }
  handleType = hipMemHandleTypePosixFileDescriptor;
#endif

  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.handleTypes = handleType;
  hipMemPool_t mempool;
  HIP_CHECK(hipMemPoolCreate(&mempool, &pool_props));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  int* A_d;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), byte_size(), mempool, stream));

  std::vector<int> A_h(DATA_SIZE());
  for (int i = 0; i < DATA_SIZE(); i++) A_h[i] = i % 1024;
  HIP_CHECK(hipMemcpyAsync(A_d, A_h.data(), byte_size(), hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  hipShareableHdl sharedHandle;
  HIP_CHECK(hipMemPoolExportToShareableHandle(&sharedHandle, mempool, handleType, 0));

  hipMemPoolPtrExportData ptrExp;
  HIP_CHECK(hipMemPoolExportPointer(&ptrExp, A_d));

  char shmName[64];
#if HT_WIN
  sprintf(shmName, "hipMemPoolIPC_shm%lu", (unsigned long)GetCurrentProcessId());
#else
  sprintf(shmName, "/hipMemPoolIPC_shm%lu", (unsigned long)getpid());
#endif

  SharedMemory shm;
  REQUIRE(shm.create(shmName, sizeof(mempoolIpcShmStruct)) == 0);
  auto *shmData = shm.as<mempoolIpcShmStruct>();
  shmData->barrier.store(0, std::memory_order_relaxed);
  shmData->sense.store(0, std::memory_order_relaxed);
  shmData->ptrExportData = ptrExp;
  shmData->handleType = handleType;
  shmData->device = 0;

  std::string exePath = getSelfExePath();
  REQUIRE(!exePath.empty());

  hip::SpawnProc child(exePath);
  REQUIRE(child.spawn("Unit_hipMemPoolExportToShareableHandle_multiproc_child") == 0);

  ipcSocketCom sockObj(true);

  barrierWait(shmData->barrier, shmData->sense, 2);

  checkSysCallErrors(sockObj.sendShareableHdl(sharedHandle, child.getProcess()));

  int exitCode = child.wait();
  REQUIRE(exitCode == 0);

  std::vector<int> B_h(DATA_SIZE());
  HIP_CHECK(hipMemcpyAsync(B_h.data(), A_d, byte_size(), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  for (int i = 0; i < DATA_SIZE(); i++) {
    REQUIRE(B_h[i] == (A_h[i] * A_h[i]));
  }

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemPoolDestroy(mempool));
  checkSysCallErrors(sockObj.closeThisSock());
#if HT_LINUX
  shm_unlink(shmName);
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Child process for multiproc test. Invoked via self-re-exec with
 * [multiproc_child] tag filter. Opens shared memory created by the parent,
 * imports the mempool handle and pointer, runs a square kernel, and exits.
 * This test is NOT meant to be run standalone; it is only invoked by the
 * parent test (Unit_hipMemPoolExportToShareableHandle_multiproc).
 * ------------------------
 *    - unit/memory/hipMemPoolExportToShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipMemPoolExportToShareableHandle_multiproc_child) {
  unsigned long parentPid = getParentProcessId();
  if (parentPid == 0) {
    HipTest::HIP_SKIP_TEST("Not launched by parent test. Skipping..");
    return;
  }
  char shmName[64];
#if HT_WIN
  sprintf(shmName, "hipMemPoolIPC_shm%lu", parentPid);
#else
  sprintf(shmName, "/hipMemPoolIPC_shm%lu", parentPid);
#endif

  SharedMemory shm;
  if (shm.open(shmName, sizeof(mempoolIpcShmStruct)) != 0) {
    HipTest::HIP_SKIP_TEST("Parent shared memory not found. "
                            "This test should only be invoked by "
                            "Unit_hipMemPoolExportToShareableHandle_multiproc. Skipping..");
    return;
  }
  auto *shmData = shm.as<mempoolIpcShmStruct>();

  hipMemPoolPtrExportData ptrExp = shmData->ptrExportData;
  hipMemAllocationHandleType handleType = shmData->handleType;
  int device = shmData->device;

  HIP_CHECK(hipSetDevice(device));

  ipcSocketCom sockObj(false);

  barrierWait(shmData->barrier, shmData->sense, 2);

  hipShareableHdl shdl;
  checkSysCallErrors(sockObj.recvShareableHdl(&shdl));

  hipMemPool_t mempoolImp;
  HIP_CHECK(hipMemPoolImportFromShareableHandle(&mempoolImp, (void *)shdl, handleType, 0));

  void *ptrImp;
  HIP_CHECK(hipMemPoolImportPointer(&ptrImp, mempoolImp, &ptrExp));

  square_kernel<<<dim3(DATA_SIZE() / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0>>>(
      (int *)ptrImp);
  HIP_CHECK(hipStreamSynchronize(0));

  checkSysCallErrors(sockObj.closeThisSock());
}

/**
 * Test Description
 * ------------------------
 *    - Negative Tests for hipMemPoolExportToShareableHandle.
 * ------------------------
 *    - unit/memory/hipMemPoolExportImportToShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolExportToShareableHandle_Negative) {
  hipShareableHdl sharedHandle;
  hipMemPoolProps pool_props{};
  hipMemPool_t mempoolPfd, mempoolwoPfd;
  checkMempoolSupported(0)
  #if HT_WIN
  hipMemAllocationHandleType handleType = hipMemHandleTypeWin32;
  #else
  hipMemAllocationHandleType handleType = hipMemHandleTypePosixFileDescriptor;
  #endif
  // Create mempool with Posix File Descriptor
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.handleTypes = handleType;
  HIP_CHECK(hipMemPoolCreate(&mempoolPfd, &pool_props));

  // Create mempool without File Descriptor
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.handleTypes = hipMemHandleTypeNone;
  HIP_CHECK(hipMemPoolCreate(&mempoolwoPfd, &pool_props));
  SECTION("Passing nullptr as handle") {
    HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(nullptr, mempoolPfd,
                                                      handleType, 0),
                    hipErrorInvalidValue);
  }
  SECTION("Passing nullptr as mempool") {
    HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(&sharedHandle, nullptr,
                                                      handleType, 0),
                    hipErrorInvalidValue);
  }
  SECTION("Passing invalid handle type") {
    HIP_CHECK_ERROR(
        hipMemPoolExportToShareableHandle(&sharedHandle, mempoolPfd, hipMemHandleTypeNone, 0),
        hipErrorInvalidValue);
  }
  SECTION("Passing mempool without file descriptor") {
    HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(&sharedHandle, mempoolwoPfd,
                                                      handleType, 0),
                    hipErrorInvalidValue);
  }
  HIP_CHECK(hipMemPoolDestroy(mempoolPfd));
  HIP_CHECK(hipMemPoolDestroy(mempoolwoPfd));
}
/**
 * End doxygen group MemoryTest.
 * @}
 */
