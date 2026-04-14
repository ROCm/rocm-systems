/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

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
TEST_CASE(Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process) {
  hipDeviceptr_t ptr1, ptr2;
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr1), 1024));
  HIP_CHECK(hipIpcGetMemHandle(&handle, reinterpret_cast<void*>(ptr1)));
  HIP_CHECK_ERROR(
      hipIpcOpenMemHandle(reinterpret_cast<void**>(&ptr2), handle, hipIpcMemLazyEnablePeerAccess),
      hipErrorInvalidContext);
  HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr1)));
}

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
TEST_CASE(Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device) {
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

/**
 * End doxygen group DeviceTest.
 * @}
 */
