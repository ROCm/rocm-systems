/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <string>
#include <thread>
#include <vector>

/**
 * @addtogroup hiprtc hiprtc
 * @{
 * @ingroup hiprtc
 * Verifies that lazy hiprtc initialization is thread-safe.
 */

namespace {
// Set by the parent test so that the child test case knows it runs in a fresh process.
constexpr auto kChildEnvVar = "HIPRTC_TEST_THREAD_SAFE_INIT_CHILD";

// HIPRTC_COMPILE_OPTIONS_APPEND is read from the environment during hiprtc initialization. A thread
// that compiles before initialization has completed misses the define and fails with #error.
constexpr auto kAppendOptionsEnvVar = "HIPRTC_COMPILE_OPTIONS_APPEND";
constexpr auto kAppendOptions = "-DHIPRTC_TEST_INIT_DONE=1";

constexpr auto kAddOneSource{R"(
#ifndef HIPRTC_TEST_INIT_DONE
#error "HIPRTC_COMPILE_OPTIONS_APPEND was not applied: compiled before hiprtc init completed"
#endif
extern "C" __global__ void add_one(int* data) { data[threadIdx.x] += HIPRTC_TEST_INIT_DONE; }
)"};

struct ThreadResult {
  std::string error;
  std::vector<char> code;
};

std::string RtcError(const char* api, hiprtcResult err) {
  return std::string(api) + " returned " + hiprtcGetErrorString(err);
}

void CompileInThread(size_t idx, const std::string& arch_option, std::atomic<size_t>& ready,
                     const std::atomic<bool>& go, ThreadResult& result) {
  ready.fetch_add(1, std::memory_order_release);
  while (!go.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  // Enter hiprtc through different APIs so that initialization is triggered from more than one
  // entry point at the same time.
  if (idx % 2 == 0) {
    int major = 0, minor = 0;
    if (const auto err = hiprtcVersion(&major, &minor); err != HIPRTC_SUCCESS) {
      result.error = RtcError("hiprtcVersion", err);
      return;
    }
  }

  hiprtcProgram prog;
  if (const auto err = hiprtcCreateProgram(&prog, kAddOneSource, "add_one.cc", 0, nullptr, nullptr);
      err != HIPRTC_SUCCESS) {
    result.error = RtcError("hiprtcCreateProgram", err);
    return;
  }

  const char* options[] = {arch_option.c_str()};
  if (const auto err = hiprtcCompileProgram(prog, 1, options); err != HIPRTC_SUCCESS) {
    result.error = RtcError("hiprtcCompileProgram", err);
    size_t log_size = 0;
    if (hiprtcGetProgramLogSize(prog, &log_size) == HIPRTC_SUCCESS && log_size) {
      std::string log(log_size, '\0');
      if (hiprtcGetProgramLog(prog, log.data()) == HIPRTC_SUCCESS) {
        result.error += "\n" + log;
      }
    }
    hiprtcDestroyProgram(&prog);
    return;
  }

  size_t code_size = 0;
  if (const auto err = hiprtcGetCodeSize(prog, &code_size); err != HIPRTC_SUCCESS) {
    result.error = RtcError("hiprtcGetCodeSize", err);
    hiprtcDestroyProgram(&prog);
    return;
  }
  result.code.resize(code_size);
  if (const auto err = hiprtcGetCode(prog, result.code.data()); err != HIPRTC_SUCCESS) {
    result.error = RtcError("hiprtcGetCode", err);
    hiprtcDestroyProgram(&prog);
    return;
  }

  if (const auto err = hiprtcDestroyProgram(&prog); err != HIPRTC_SUCCESS) {
    result.error = RtcError("hiprtcDestroyProgram", err);
  }
}
}  // anonymous namespace

/**
 * Test Description
 * ------------------------
 *  - Runs Unit_hiprtc_ThreadSafeInit_Child several times, each time in a fresh process, so that
 *    every run starts with hiprtc uninitialized and its first calls race each other.
 * Test source
 * ------------------------
 *  - unit/rtc/hiprtcThreadSafeInit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hiprtc_ThreadSafeInit) {
  // The race window is small, give it several chances to show up.
  constexpr int kIterations = 5;
  for (int i = 0; i < kIterations; ++i) {
    hip::SpawnProc proc(getSelfExePath(), true, true);
    proc.setEnv(kChildEnvVar, "1");
    proc.setEnv(kAppendOptionsEnvVar, kAppendOptions);
    // Make sure that every thread really compiles instead of reading the result from the cache.
    proc.setEnv("AMD_COMGR_CACHE", "0");
    const int exit_code = proc.run("Unit_hiprtc_ThreadSafeInit_Child");
    INFO("Iteration: " << i << "\nChild output:\n" << proc.getOutput());
    REQUIRE(exit_code == 0);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Child of Unit_hiprtc_ThreadSafeInit, not meant to run on its own.
 *  - Makes the very first hiprtc calls of the process concurrently from multiple threads, compiles a
 *    kernel in every thread and verifies that every produced code object can be loaded and run.
 *  - The kernel only compiles if HIPRTC_COMPILE_OPTIONS_APPEND has been applied, which detects
 *    threads that start compiling before hiprtc initialization has completed.
 * Test source
 * ------------------------
 *  - unit/rtc/hiprtcThreadSafeInit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hiprtc_ThreadSafeInit_Child) {
  if (std::getenv(kChildEnvVar) == nullptr) {
    HIP_SKIP_TEST("Only run by Unit_hiprtc_ThreadSafeInit.");
  }

  // No hiprtc API may be called before the worker threads are released.
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  const std::string arch_option = std::string("--gpu-architecture=") + props.gcnArchName;

  const size_t thread_count = std::clamp(std::thread::hardware_concurrency(), 4u, 16u);
  std::vector<ThreadResult> results(thread_count);
  std::vector<std::thread> threads;
  std::atomic<size_t> ready{0};
  std::atomic<bool> go{false};

  for (size_t i = 0; i < thread_count; ++i) {
    threads.emplace_back(CompileInThread, i, std::cref(arch_option), std::ref(ready),
                         std::cref(go), std::ref(results[i]));
  }
  // Release all threads at once to maximize contention on initialization.
  while (ready.load(std::memory_order_acquire) != thread_count) {
    std::this_thread::yield();
  }
  go.store(true, std::memory_order_release);
  for (auto& t : threads) {
    t.join();
  }

  // Catch2 assertions are not thread-safe, so results are only checked here.
  for (size_t i = 0; i < thread_count; ++i) {
    INFO("Thread: " << i << "\n" << results[i].error);
    REQUIRE(results[i].error.empty());
    REQUIRE(!results[i].code.empty());
  }

  // Every code object must be usable: each one increments the buffer once.
  constexpr int kElements = 64;
  int* data_d;
  HIP_CHECK(hipMalloc(&data_d, kElements * sizeof(int)));
  HIP_CHECK(hipMemset(data_d, 0, kElements * sizeof(int)));
  for (const auto& result : results) {
    hipModule_t module;
    hipFunction_t kernel;
    HIP_CHECK(hipModuleLoadData(&module, result.code.data()));
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "add_one"));
    void* args[] = {&data_d};
    HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, kElements, 1, 1, 0, nullptr, args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipModuleUnload(module));
  }

  std::vector<int> data_h(kElements);
  HIP_CHECK(hipMemcpy(data_h.data(), data_d, kElements * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(data_d));
  REQUIRE(std::all_of(data_h.begin(), data_h.end(),
                      [&](int v) { return v == static_cast<int>(thread_count); }));
}

/**
 * End doxygen group hiprtc.
 * @}
 */
