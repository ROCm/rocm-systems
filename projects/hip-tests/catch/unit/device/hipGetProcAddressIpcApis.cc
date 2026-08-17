/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_process.hh>
#include <utils.hh>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>
#include "hipGetProcAddressHelpers.hh"

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - memory IPC related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress_IPC_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipGetProcAddress_IPC_Memory) {
  int N = 40;
  int Nbytes = N * sizeof(int);

  int fd[2];
  REQUIRE(pipe(fd) == 0);

  auto pid = fork();

  // Validating hipIpcGetMemHandle API
  if (pid != 0) {  // parent process
    void* hipIpcGetMemHandle_ptr = nullptr;

    int currentHipVersion = 0;
    HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

    HIP_CHECK(hipGetProcAddress("hipIpcGetMemHandle", &hipIpcGetMemHandle_ptr, currentHipVersion, 0,
                                nullptr));

    hipError_t (*dyn_hipIpcGetMemHandle_ptr)(hipIpcMemHandle_t*, void*) =
        reinterpret_cast<hipError_t (*)(hipIpcMemHandle_t*, void*)>(hipIpcGetMemHandle_ptr);

    int* srcHostMem = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(srcHostMem != nullptr);
    fillHostArray(srcHostMem, N, 10);

    int* devMemSrc = nullptr;
    HIP_CHECK(hipMalloc(&devMemSrc, Nbytes));
    REQUIRE(devMemSrc != nullptr);
    HIP_CHECK(hipMemcpy(devMemSrc, srcHostMem, Nbytes, hipMemcpyHostToDevice));

    hipIpcMemHandle_t handle;
    HIP_CHECK(dyn_hipIpcGetMemHandle_ptr(&handle, devMemSrc));

    REQUIRE(hip::writeAll(fd[1], &handle, sizeof(handle)));
    REQUIRE(close(fd[1]) == 0);

    REQUIRE(wait(NULL) >= 0);

    HIP_CHECK(hipFree(devMemSrc));
    free(srcHostMem);
  } else {  // child process
    // Validating hipIpcOpenMemHandle, hipIpcCloseMemHandle API's
    void* hipIpcOpenMemHandle_ptr = nullptr;
    void* hipIpcCloseMemHandle_ptr = nullptr;

    int currentHipVersion = 0;
    HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

    HIP_CHECK(hipGetProcAddress("hipIpcOpenMemHandle", &hipIpcOpenMemHandle_ptr, currentHipVersion,
                                0, nullptr));
    HIP_CHECK(hipGetProcAddress("hipIpcCloseMemHandle", &hipIpcCloseMemHandle_ptr,
                                currentHipVersion, 0, nullptr));

    hipError_t (*dyn_hipIpcOpenMemHandle_ptr)(void**, hipIpcMemHandle_t, unsigned int) =
        reinterpret_cast<hipError_t (*)(void**, hipIpcMemHandle_t, unsigned int)>(
            hipIpcOpenMemHandle_ptr);
    hipError_t (*dyn_hipIpcCloseMemHandle_ptr)(void*) =
        reinterpret_cast<hipError_t (*)(void*)>(hipIpcCloseMemHandle_ptr);

    hipIpcMemHandle_t handle;
    REQUIRE(hip::readAll(fd[0], &handle, sizeof(handle)));
    REQUIRE(close(fd[0]) == 0);

    int* devPtr = nullptr;
    HIP_CHECK(dyn_hipIpcOpenMemHandle_ptr(reinterpret_cast<void**>(&devPtr), handle,
                                          hipIpcMemLazyEnablePeerAccess));
    REQUIRE(devPtr != nullptr);

    addOneKernel<<<1, 1>>>(devPtr, N);

    int* dstHostMem = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(dstHostMem != nullptr);

    HIP_CHECK(hipMemcpy(dstHostMem, devPtr, Nbytes, hipMemcpyDeviceToHost));
    REQUIRE(validateHostArray(dstHostMem, N, 11) == true);

    HIP_CHECK(dyn_hipIpcCloseMemHandle_ptr(devPtr));

    free(dstHostMem);
  }
}

namespace {

void runIpcEventProcAddressTest(std::optional<int> parentDeviceId,
                                std::optional<int> childDeviceId) {
  int fd[2];
  REQUIRE(pipe(fd) == 0);

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid != 0) {  // parent process: exports the interprocess event handle
    if (parentDeviceId) {
      HIP_CHECK(hipSetDevice(*parentDeviceId));
    }

    void* hipIpcGetEventHandle_ptr = nullptr;

    int currentHipVersion = 0;
    HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

    HIP_CHECK(hipGetProcAddress("hipIpcGetEventHandle", &hipIpcGetEventHandle_ptr,
                                currentHipVersion, 0, nullptr));
    REQUIRE(hipIpcGetEventHandle_ptr != nullptr);

    auto dyn_hipIpcGetEventHandle_ptr =
        reinterpret_cast<hipError_t (*)(hipIpcEventHandle_t*, hipEvent_t)>(
            hipIpcGetEventHandle_ptr);

    // Interprocess events must disable timing.
    hipEvent_t event = nullptr;
    HIP_CHECK(hipEventCreateWithFlags(&event, hipEventInterprocess | hipEventDisableTiming));
    REQUIRE(event != nullptr);

    hipIpcEventHandle_t handle{};
    HIP_CHECK(dyn_hipIpcGetEventHandle_ptr(&handle, event));

    // The parent only writes to the pipe.
    REQUIRE(close(fd[0]) == 0);

    REQUIRE(hip::writeAll(fd[1], &handle, sizeof(handle)));
    REQUIRE(close(fd[1]) == 0);

    // Surface any consumer-side failure (a forked child's REQUIREs do not
    // propagate to this process).
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == EXIT_SUCCESS);

    HIP_CHECK(hipEventDestroy(event));

  } else { // child process: imports the event handle in a separate process
    bool ok = false;
    try {
      void* hipIpcOpenEventHandle_ptr = nullptr;

      int currentHipVersion = 0;
      HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

      HIP_CHECK(hipGetProcAddress("hipIpcOpenEventHandle", &hipIpcOpenEventHandle_ptr,
                                  currentHipVersion, 0, nullptr));
      REQUIRE(hipIpcOpenEventHandle_ptr != nullptr);

      auto dyn_hipIpcOpenEventHandle_ptr =
          reinterpret_cast<hipError_t (*)(hipEvent_t*, hipIpcEventHandle_t)>(
              hipIpcOpenEventHandle_ptr);

      // The child only reads from the pipe.
      REQUIRE(close(fd[1]) == 0);

      hipIpcEventHandle_t handle{};
      REQUIRE(hip::readAll(fd[0], &handle, sizeof(handle)));
      REQUIRE(close(fd[0]) == 0);

      if (childDeviceId) {
        HIP_CHECK(hipSetDevice(*childDeviceId));
      }

      // The call under test: import the handle exported by the parent through the
      // proc-address-resolved pointer.
      hipEvent_t event = nullptr;
      HIP_CHECK(dyn_hipIpcOpenEventHandle_ptr(&event, handle));
      REQUIRE(event != nullptr);

      // Exercise the imported event in a real GPU workload to confirm it is a
      // usable event object. All work is issued on a single stream and the
      // imported event is used as the completion sync point.
      constexpr int N = 40;
      constexpr int Nbytes = N * sizeof(int);

      hipStream_t stream = nullptr;
      HIP_CHECK(hipStreamCreate(&stream));

      std::vector<int> hostMem(N, 10);

      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);

      HIP_CHECK(hipMemcpyAsync(devMem, hostMem.data(), Nbytes, hipMemcpyHostToDevice, stream));
      addOneKernel<<<1, 1, 0, stream>>>(devMem, N);
      HIP_CHECK(hipMemcpyAsync(hostMem.data(), devMem, Nbytes, hipMemcpyDeviceToHost, stream));

      // Record the imported event on the same stream and wait on it. The wait
      // covers the whole pipeline above, so the results are ready to validate and
      // the buffer is safe to read once it returns.
      HIP_CHECK(hipEventRecord(event, stream));
      HIP_CHECK(hipEventSynchronize(event));

      REQUIRE(validateHostArray(hostMem.data(), N, 11));

      HIP_CHECK(hipFree(devMem));
      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipEventDestroy(event));
      ok = true;
    } catch (const std::exception& e) {
      // A forked child's REQUIRE/HIP_CHECK failures do not reach the Catch2
      // reporter, so log the reason to stderr before signalling it through the
      // exit code -- otherwise CI only shows an opaque "Subprocess aborted".
      fprintf(stderr, "[IPC_Event child] failure: %s\n", e.what());
      _exit(EXIT_FAILURE);
    } catch (...) {
      fprintf(stderr, "[IPC_Event child] failure: unknown exception\n");
      _exit(EXIT_FAILURE);
    }
    _exit(ok ? EXIT_SUCCESS : EXIT_FAILURE);
  }
}
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Verifies that hipGetProcAddress returns usable function pointers for the
 *  - Event IPC APIs (hipIpcGetEventHandle / hipIpcOpenEventHandle): the parent
 *  - exports an interprocess event handle through the resolved
 *  - hipIpcGetEventHandle pointer, and the child imports it through the resolved
 *  - hipIpcOpenEventHandle pointer in a separate process.
 *  - This test's unique purpose is to exercise the proc-address dispatch path for
 *  - these APIs (address + ABI). Full cross-process event *synchronization*
 *  - semantics are covered by Unit_hipIpcEventHandle_Functional.
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress_IPC_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipGetProcAddress_IPC_Event) {
  runIpcEventProcAddressTest(std::nullopt, std::nullopt);
}

/**
 * Test Description
 * ------------------------
 *  - Runs the Event IPC proc-address path with both processes on device 1
 *  - through hipSetDevice, so the imported event is recorded on a GPU that is
 *  - not the first agent of the topology and its interprocess signal page has to
 *  - be reachable from there.
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress_IPC_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 *  - Multiple devices
 */
HIP_TEST_CASE(Unit_hipGetProcAddress_IPC_Event_SetDevice) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  runIpcEventProcAddressTest(1, 1);
}

/**
 * Test Description
 * ------------------------
 *  - Runs the Event IPC proc-address path with the parent on device 0 and the
 *  - child on device 1, so the imported interprocess signal page exported from
 *  - one GPU must be reachable when the consumer records on another.
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress_IPC_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 *  - Multiple devices
 */
HIP_TEST_CASE(Unit_hipGetProcAddress_IPC_Event_CrossDevice) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  runIpcEventProcAddressTest(0, 1);
}
