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

// The reset is gated by HIP runtime initialization only on Linux, to protect
// fork-based multiprocess parents; /proc and the GPU device nodes exist there.
#if HT_LINUX
#include <dirent.h>
#include <limits.h>
#include <unistd.h>
#include <cstring>

// The GPU character devices a HIP runtime opens while initializing. Matched as
// prefixes, so the per-device suffixes (renderD128, nvidia0, ...) are covered.
static constexpr const char* kGpuDeviceNodes[] = {
#if HT_AMD
    "/dev/kfd", "/dev/dri/renderD",
#elif HT_NVIDIA
    "/dev/nvidia",
#else
#error "Platform not recognized"
#endif
};

// Whether this process has initialized the HIP runtime, hence may hold a leaked
// last error. HIP cannot be asked -- every entry point that knows runs
// HIP_INIT_API first -- so look for the device nodes it opens instead: no HIP
// call runs without one, and touching no backend keeps this header-free.
static bool hipRuntimeInitialized() {
  // HIP is never de-initialized mid-run, so this only ever flips false -> true.
  static bool initialized = false;
  if (initialized) {
    return true;
  }

  DIR* dir = opendir("/proc/self/fd");
  if (dir == nullptr) {
    return false;
  }
  // Resolving each entry against the directory fd keeps this free of path
  // formatting, which -Wformat-truncation rejects for a 255-byte d_name, and
  // saves the kernel re-walking /proc/self/fd for every entry.
  const int dir_fd = dirfd(dir);
  while (const dirent* entry = readdir(dir)) {
    char target[PATH_MAX];
    const ssize_t length = readlinkat(dir_fd, entry->d_name, target, sizeof(target) - 1);
    if (length <= 0) {
      continue;
    }
    target[length] = '\0';
    for (const char* node : kGpuDeviceNodes) {
      if (strncmp(target, node, strlen(node)) == 0) {
        initialized = true;
        break;
      }
    }
    if (initialized) {
      break;
    }
  }
  closedir(dir);
  return initialized;
}
#endif  // HT_LINUX

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
