/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

HIP_TEST_CASE(Unit_hipMemGetInfo_FreeLessThanTotal) {
  unsigned int* A_mem{nullptr};
  size_t freeMemInit, totalMemInit;
  size_t freeMem, totalMem;

  HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));
  REQUIRE(freeMemInit <= totalMemInit);
  HIP_CHECK(hipMalloc(&A_mem, 1024));
  HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));
  REQUIRE(freeMem < totalMem);
  REQUIRE(totalMem == totalMemInit);

  HIP_CHECK(hipFree(A_mem));
}

#if HT_WIN
#include <hip_test_process.hh>

#define MEMGETINFO_MPROC_TAG "HIP_MEMGETINFO_MPROC_TAG"

static std::string memGetInfoEventName(const std::string& tag, const char* what) {
  return "hipMemGetInfoMProc_" + tag + "_" + what;
}

/**
 * Spawn a child process that allocates 2 GB of device memory and signals the
 * parent. Verify from the parent process that the 2 GB held by the other process
 * is not reported as free. Windows counterpart of the fork() based
 * Unit_hipMemGetInfo_Functional_Scenario2 in multiproc/hipMemGetInfoMProc.cc.
 */
HIP_TEST_CASE(Unit_hipMemGetInfo_Functional_OtherProcess) {
  constexpr size_t size = 2147483648;  // 2GB
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  if (prop.integrated) {
    HIP_SKIP_TEST("Free memory of an integrated GPU is not device-wide on Windows.");
  }
  size_t free = 0, total = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  if (free < 2 * size) {
    HIP_SKIP_TEST("Not enough free device memory.");
  }

  const std::string tag = std::to_string(GetCurrentProcessId());
  HANDLE ready = CreateEventA(nullptr, TRUE, FALSE, memGetInfoEventName(tag, "ready").c_str());
  HANDLE done = CreateEventA(nullptr, TRUE, FALSE, memGetInfoEventName(tag, "done").c_str());
  REQUIRE(ready != nullptr);
  REQUIRE(done != nullptr);

  hip::SpawnProc child(getSelfExePath());
  child.setEnv(MEMGETINFO_MPROC_TAG, tag);
  REQUIRE(child.spawn("Unit_hipMemGetInfo_Functional_OtherProcess_Child") == 0);
  // Wait until the child holds the memory (or exits early), then check it from this process
  HANDLE waitFor[] = {ready, child.getProcess().hProcess};
  const bool childReady = (WaitForMultipleObjects(2, waitFor, FALSE, 60000) == WAIT_OBJECT_0);
  if (childReady) {
    HIP_CHECK(hipMemGetInfo(&free, &total));
  }
  // Let the child free the memory and exit
  SetEvent(done);
  const int exitCode = child.wait();
  CloseHandle(ready);
  CloseHandle(done);

  REQUIRE(childReady);
  REQUIRE(exitCode == 0);
  REQUIRE((total - free) >= size);
}

/**
 * Child process of Unit_hipMemGetInfo_Functional_OtherProcess: allocates 2 GB
 * of device memory, signals the parent and holds the memory until the parent
 * is done. Skipped when not launched by the parent test.
 */
HIP_TEST_CASE(Unit_hipMemGetInfo_Functional_OtherProcess_Child) {
  constexpr size_t size = 2147483648;  // 2GB
  const char* tag = std::getenv(MEMGETINFO_MPROC_TAG);
  if (tag == nullptr || tag[0] == '\0') {
    HIP_SKIP_TEST("This test must be launched by parent multiprocess test.");
  }
  HANDLE ready = OpenEventA(EVENT_MODIFY_STATE, FALSE, memGetInfoEventName(tag, "ready").c_str());
  HANDLE done = OpenEventA(SYNCHRONIZE, FALSE, memGetInfoEventName(tag, "done").c_str());
  REQUIRE(ready != nullptr);
  REQUIRE(done != nullptr);

  char* A_d = nullptr;
  HIP_CHECK(hipMalloc(&A_d, size));
  HIP_CHECK(hipMemset(A_d, 0, size));
  HIP_CHECK(hipDeviceSynchronize());
  // Signal the parent and wait until it has checked the memory
  SetEvent(ready);
  WaitForSingleObject(done, 60000);
  HIP_CHECK(hipFree(A_d));
  CloseHandle(ready);
  CloseHandle(done);
}
#endif
