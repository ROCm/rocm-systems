/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_EXPECTEDCRASH_H_
#define RCCL_TEST_HOST_EXPECTEDCRASH_H_

// Shared by the death tests that expect a SIGSEGV.

#include <gtest/gtest.h>

#if defined(__linux__)
#include <sys/prctl.h>
#endif
#include <sys/resource.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <string>

// ASan turns a SIGSEGV into a report plus exit(1), so KilledBySignal never matches under it; match the exit instead.
#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
#define DEATH_BY_SEGV ::testing::ExitedWithCode(1)
#else
#define DEATH_BY_SEGV ::testing::KilledBySignal(SIGSEGV)
#endif

// These tests deliberately raise fatal signals. Some CI environments enable core dumps, and a
// shuffled death test can inherit a large address space from an earlier test. Dumping that address
// space turns a millisecond assertion into a multi-second test. PR_SET_DUMPABLE covers both regular
// core files and piped collectors (for which Linux ignores RLIMIT_CORE); retain the limit as defense
// in depth. Apply both inside the death statement so only gtest's child is affected and unexpected
// parent-process crashes remain dumpable.
inline void DisableCoreDumpsForExpectedCrash() {
#if defined(__linux__)
  if (prctl(PR_SET_DUMPABLE, 0) != 0) {
    std::perror("prctl(PR_SET_DUMPABLE) failed");
    _exit(1);
  }
#endif
  const struct rlimit noCore = {0, 0};
  if (setrlimit(RLIMIT_CORE, &noCore) != 0) {
    std::perror("setrlimit(RLIMIT_CORE) failed");
    _exit(1);
  }
}

// death_test_style is process-wide; restore it so a shuffled run doesn't depend on which test ran first.
class ScopedDeathTestStyle {
 public:
  explicit ScopedDeathTestStyle(const char* style) : saved_(::testing::GTEST_FLAG(death_test_style)) {
    ::testing::GTEST_FLAG(death_test_style) = style;
  }
  ~ScopedDeathTestStyle() { ::testing::GTEST_FLAG(death_test_style) = saved_; }
  ScopedDeathTestStyle(const ScopedDeathTestStyle&) = delete;
  ScopedDeathTestStyle& operator=(const ScopedDeathTestStyle&) = delete;

 private:
  std::string saved_;
};

#endif  // RCCL_TEST_HOST_EXPECTEDCRASH_H_
