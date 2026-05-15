/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

/**
 * @addtogroup hipIpcOpenMemHandle hipIpcOpenMemHandle
 * @{
 * @ingroup DeviceTest
 * `hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags)` -
 * Opens an interprocess memory handle exported from another process
 * and returns a device pointer usable in the local process.
 */

/**
 * Test Description
 * ------------------------
 *  - Handle the attempt to open memory handle in the same process
 *    that has created it.
 *      -# When the process is the same
 *        - Expected output: return `hipErrorInvalidContext`
 * Test source
 * ------------------------
 *  - unit/device/hipIpcOpenMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process) {
  hipDeviceptr_t ptr1, ptr2;
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr1), 1024));
  HIP_CHECK(hipIpcGetMemHandle(&handle, reinterpret_cast<void*>(ptr1)));
  HIP_CHECK_ERROR(
      hipIpcOpenMemHandle(reinterpret_cast<void**>(&ptr2), handle, hipIpcMemLazyEnablePeerAccess),
      hipErrorInvalidContext);
  HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr1)));
}


#if HT_LINUX
/**
 * Test Description
 * ------------------------
 *  - Checks that opening the same memory handle from a different context
 *    returns error
 *    -# When different context
 *      - Expected output: return `hipErrorInvalidResourceHandle`
 * Test source
 * ------------------------
 *  - unit/device/hipIpcOpenMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device) {
  int fd[2];
  REQUIRE(pipe(fd) == 0);

  // The fork must be performed before the runtime is initialized(so before any API that implicitly
  // initializes it). The pipe in conjunction with wait is then used to impose total ordering
  // between parent and child process. Because total ordering is imposed regular CATCH assertions
  // should be safe to use
  auto pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {  // child
    REQUIRE(close(fd[1]) == 0);

    hipIpcMemHandle_t handle;
    REQUIRE(read(fd[0], &handle, sizeof(handle)) >= 0);
    REQUIRE(close(fd[0]) == 0);

    hipDeviceptr_t ptr_child;
    HIP_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void**>(&ptr_child), handle,
                                  hipIpcMemLazyEnablePeerAccess));

    HIP_CHECK(hipInit(0));
    hipCtx_t ctx;
    HIP_CHECK(hipCtxCreate(&ctx, 0, 0));

    hipDeviceptr_t ptr_child_ctx;
    HIP_CHECK_ERROR(hipIpcOpenMemHandle(reinterpret_cast<void**>(&ptr_child_ctx), handle,
                                        hipIpcMemLazyEnablePeerAccess),
                    hipErrorInvalidResourceHandle);

    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);

    hipDeviceptr_t ptr;
    hipIpcMemHandle_t handle;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), 1024));
    HIP_CHECK(hipIpcGetMemHandle(&handle, reinterpret_cast<void*>(ptr)));

    REQUIRE(write(fd[1], &handle, sizeof(handle)) >= 0);
    REQUIRE(close(fd[1]) == 0);

    REQUIRE(wait(NULL) >= 0);

    HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr)));
  }
}
#endif

constexpr int DATA_SIZE = 1024 * 1024;
constexpr size_t BYTE_SIZE = DATA_SIZE * sizeof(int);
constexpr int THREADS_PER_BLOCK = 512;

static __global__ void square_kernel(int* buf) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  buf[i] = buf[i] * buf[i];
}

static std::string ipcHandleToHex(const hipIpcMemHandle_t& handle) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (size_t i = 0; i < sizeof(handle.reserved); i++) {
    oss << std::setw(2)
        << static_cast<unsigned>(static_cast<unsigned char>(handle.reserved[i]));
  }
  return oss.str();
}

static hipIpcMemHandle_t hexToIpcHandle(const std::string& hex_str) {
  hipIpcMemHandle_t handle = {};
  std::stringstream hex_ss(hex_str);
  for (size_t i = 0; i < sizeof(handle.reserved); i++) {
    std::string byte_str(2, '\0');
    if (!hex_ss.read(&byte_str[0], 2)) break;
    unsigned int byte_val = 0;
    std::stringstream byte_ss;
    byte_ss << std::hex << byte_str;
    byte_ss >> byte_val;
    handle.reserved[i] = static_cast<char>(static_cast<unsigned char>(byte_val));
  }
  return handle;
}

/**
 * Test Description
 * ------------------------
 *  - Parent process: allocates device memory, fills with i%1024,
 *    exports the IPC handle via environment variable, spawns a child
 *    process that squares the data through the shared handle, then
 *    validates the result.
 * Test source
 * ------------------------
 *  - unit/device/hipIpcOpenMemHandle.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcOpenMemHandle_Multiproc) {
  int* d_buf;
  HIP_CHECK(hipMalloc(&d_buf, BYTE_SIZE));

  std::vector<int> h_buf(DATA_SIZE);
  for (int i = 0; i < DATA_SIZE; i++) {
    h_buf[i] = i % 1024;
  }
  HIP_CHECK(hipMemcpy(d_buf, h_buf.data(), BYTE_SIZE, hipMemcpyHostToDevice));

  hipIpcMemHandle_t handle;
  HIP_CHECK(hipIpcGetMemHandle(&handle, d_buf));

  std::string hex = ipcHandleToHex(handle);

  hip::SpawnProc child(getSelfExePath());
  child.setEnv("HIP_IPC_HANDLE", hex);
  REQUIRE(child.spawn("Unit_hipIpcOpenMemHandle_Multiproc_Child") == 0);
  REQUIRE(child.wait() == 0);

  HIP_CHECK(hipMemcpy(h_buf.data(), d_buf, BYTE_SIZE, hipMemcpyDeviceToHost));
  for (int i = 0; i < DATA_SIZE; i++) {
    int expected = (i % 1024) * (i % 1024);
    REQUIRE(h_buf[i] == expected);
  }

  HIP_CHECK(hipFree(d_buf));
}

/**
 * Test Description
 * ------------------------
 *  - Child process: reads the IPC handle from the HIP_IPC_HANDLE
 *    environment variable, opens the shared device memory, squares
 *    each element via a kernel, validates the result, then closes
 *    the handle. Skipped when not launched by the parent test.
 * Test source
 * ------------------------
 *  - unit/device/hipIpcOpenMemHandle.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcOpenMemHandle_Multiproc_Child) {
  const char* hex = std::getenv("HIP_IPC_HANDLE");
  if (hex == nullptr || hex[0] == '\0') {
    HIP_SKIP_TEST("This test must be launched by parent multiprocess test.");
  }

  hipIpcMemHandle_t handle = hexToIpcHandle(hex);

  void* devPtr = nullptr;
  HIP_CHECK(hipIpcOpenMemHandle(&devPtr, handle, hipIpcMemLazyEnablePeerAccess));

  square_kernel<<<dim3(DATA_SIZE / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK)>>>(
      reinterpret_cast<int*>(devPtr));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> result(DATA_SIZE);
  HIP_CHECK(hipMemcpy(result.data(), devPtr, BYTE_SIZE, hipMemcpyDeviceToHost));
  for (int i = 0; i < DATA_SIZE; i++) {
    int expected = (i % 1024) * (i % 1024);
    REQUIRE(result[i] == expected);
  }

  HIP_CHECK(hipIpcCloseMemHandle(devPtr));
}

/**
 * End doxygen group DeviceTest.
 * @}
 */
