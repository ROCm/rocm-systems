/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <hip_test_params.hh>
#include <hip_test_context.hh>
#include <hip_test_level.hh>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

/**
 * @brief Loads the active level's test parameters into TestParameterStore.
 *
 * The level only picks parameters; it never narrows the run, so the Catch2 test
 * spec is not consulted. Catch2 never starts a run for --help or the --list-*
 * options, so those work whatever HIP_TEST_LEVEL holds.
 */
class HipTestParameterListener : public Catch::EventListenerBase {
 public:
  using Catch::EventListenerBase::EventListenerBase;

  /**
   * @brief Called once when the test run begins, before any test.
   * Initializes TestParameterStore and loads the resolved level's parameters.
   * @throws std::runtime_error when HIP_TEST_LEVEL is rejected. Catch2 prints
   *         the message and Session::run() returns 1 without running any test,
   *         rather than silently running with the default parameters.
   */
  void testRunStarting(Catch::TestRunInfo const&) override {
    const auto resolution = HipTestLevel::resolveLevel(std::getenv("HIP_TEST_LEVEL"));
    if (resolution.rejected()) {
      throw std::runtime_error("[Test Level] ERROR: " + resolution.error + ". Aborting.");
    }
    LogPrintf("%s\n", resolution.describe().c_str());

    auto& params = TestParameterStore::instance();
    params.initialize();
    params.loadLevelConfig(resolution.level);
  }

  /// @brief Called when the test run ends. Releases the loaded parameters.
  void testRunEnded(Catch::TestRunStats const&) override { TestParameterStore::instance().clear(); }
};

// Register the listener - it will be automatically activated
CATCH_REGISTER_LISTENER(HipTestParameterListener)

#if HT_AMD
extern "C" hipError_t hipExtGetLastError(void);

// The runtime calls __hipOnError on every HIP API call that returns a real
// error. Interposing it (ELF symbol interposition; Linux only) records that a
// HIP error occurred the moment any call fails, so the listener resets leaked
// errors from direct HIP calls (REQUIRE, bare statements) that bypass the check
// macros, while a parent that never calls HIP never initializes it.
extern "C" void __hipOnError(const void* err_info) {
  static_cast<void>(err_info);
  g_hipErrorWasSet.store(true, std::memory_order_relaxed);
}
#elif HT_NVIDIA
#include <cuda.h>
#else
#error "Platform not recognized"
#endif

// Whether this process has initialized the HIP runtime, hence may hold a leaked
// last error and can be reset without newly initializing a fork parent that
// never used HIP. AMD: set by the interposed __hipOnError when any call errors.
// NVIDIA: CUDA driver calls return CUDA_ERROR_NOT_INITIALIZED until the runtime
// is initialized, without initializing it -- a side-effect-free probe.
static bool hipRuntimeInitialized() {
#if HT_AMD
  return g_hipErrorWasSet.load(std::memory_order_relaxed);
#elif HT_NVIDIA
  int count = 0;
  return cuDeviceGetCount(&count) != CUDA_ERROR_NOT_INITIALIZED;
#else
#error "Platform not recognized"
#endif
}

/**
 * @brief Clears the sticky thread-local HIP last error after each test case so
 * an error left unconsumed by one test cannot fail the next. Clears both the
 * standard (hipGetLastError) and, on AMD, the command (hipExtGetLastError)
 * states. On Linux it resets only once the HIP runtime is initialized, so it
 * never initializes HIP in a parent that never called it, e.g. a fork-based
 * multiprocess parent; such parents do not exist on other platforms, which
 * always reset. A leftover error is always reported on stderr;
 * HIP_TEST_STRICT_LAST_ERROR additionally fails the run.
 */
class HipLastErrorResetListener : public Catch::EventListenerBase {
 public:
  using Catch::EventListenerBase::EventListenerBase;

  void testCaseEnded(Catch::TestCaseStats const& stats) override {
#if HT_LINUX
    if (!hipRuntimeInitialized()) {
      return;
    }
#endif
    hipError_t leftover = hipGetLastError();
#if HT_AMD
    hipError_t cmdLeftover = hipExtGetLastError();
#elif HT_NVIDIA
    hipError_t cmdLeftover = hipSuccess;
#else
#error "Platform not recognized"
#endif
    bool leaked = false;
    if (leftover != hipSuccess) {
      report(stats.testInfo->name, "last error", leftover);
      leaked = true;
    }
    if (cmdLeftover != hipSuccess && cmdLeftover != leftover) {
      report(stats.testInfo->name, "command error", cmdLeftover);
      leaked = true;
    }
    if (leaked && strict_) {
      culprits_ += "\n  " + stats.testInfo->name;
    }
  }

  void testRunEnded(Catch::TestRunStats const&) override {
    if (!culprits_.empty()) {
      fprintf(stderr, "[HIP last-error] strict mode, tests left unconsumed errors:%s\n",
              culprits_.c_str());
      g_hipStrictLastErrorFailed.store(true, std::memory_order_relaxed);
    }
  }

 private:
  static void report(std::string const& test, const char* which, hipError_t err) {
    fprintf(stderr, "[HIP last-error] test '%s' left %s %d (%s)\n", test.c_str(), which,
            static_cast<int>(err), hipGetErrorString(err));
  }

  bool strict_ = !TestContext::getEnvVar("HIP_TEST_STRICT_LAST_ERROR").empty();
  std::string culprits_;
};

CATCH_REGISTER_LISTENER(HipLastErrorResetListener)
