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
#endif

// An error produced by the reset's own hipGetLastError/hipExtGetLastError when
// the runtime is unusable (e.g. no visible device) rather than left by the test.
static bool isResetArtifact(hipError_t err) {
  return err == hipErrorNoDevice || err == hipErrorNotInitialized;
}

/**
 * @brief Clears the sticky thread-local HIP last error after each test case so
 * an error left unconsumed by one test cannot fail the next. Clears both the
 * standard (hipGetLastError) and, on AMD, the command (hipExtGetLastError)
 * states. Resets only once the process has used HIP (g_hipRuntimeInitialized),
 * so it never initializes HIP in a parent that never used it, e.g. a fork-based
 * multiprocess parent. A leftover error is always reported on stderr;
 * HIP_TEST_STRICT_LAST_ERROR additionally fails the run.
 */
class HipLastErrorResetListener : public Catch::EventListenerBase {
 public:
  using Catch::EventListenerBase::EventListenerBase;

  void testCaseEnded(Catch::TestCaseStats const& stats) override {
    if (!g_hipRuntimeInitialized.load(std::memory_order_relaxed)) {
      return;
    }
    hipError_t leftover = hipGetLastError();
#if HT_AMD
    hipError_t cmdLeftover = hipExtGetLastError();
#else
    hipError_t cmdLeftover = hipSuccess;
#endif
    bool leaked = false;
    if (leftover != hipSuccess && !isResetArtifact(leftover)) {
      report(stats.testInfo->name, "last error", leftover);
      leaked = true;
    }
    if (cmdLeftover != hipSuccess && cmdLeftover != leftover && !isResetArtifact(cmdLeftover)) {
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
