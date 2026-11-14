// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/host_access_guard.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

TEST(HostAccessGuardTest, ValidAccessRunsOnce) {
  const std::array<unsigned char, 4> source{1, 2, 3, 4};
  std::array<unsigned char, 4> destination{};
  unsigned calls = 0;
  bool armed_during_access = false;
  EXPECT_TRUE(rocjitsu::with_host_access_guard([&] {
    ++calls;
    armed_during_access = rocjitsu::host_access_guard_state().armed;
    std::memcpy(destination.data(), source.data(), source.size());
  }));
  EXPECT_EQ(calls, 1u);
  EXPECT_EQ(destination, source);
  EXPECT_FALSE(rocjitsu::host_access_guard_state().armed);
#if defined(RJ_DISABLE_HOST_FAULT_RECOVERY)
  EXPECT_FALSE(armed_during_access);
#else
  EXPECT_TRUE(armed_during_access);
#endif
}

#if defined(RJ_DISABLE_HOST_FAULT_RECOVERY)
TEST(HostAccessGuardDeathTest, DisabledRecoveryDeliversRealFaultsToTheHost) {
  for (const int expected_signal : {SIGSEGV, SIGBUS}) {
    SCOPED_TRACE(expected_signal);
    EXPECT_EXIT(([&] {
                  struct sigaction action {};
                  action.sa_handler = [](int signal_number) { _exit(64 + signal_number); };
                  if (sigemptyset(&action.sa_mask) != 0 ||
                      sigaction(expected_signal, &action, nullptr) != 0)
                    _exit(1);
                  rocjitsu::install_host_access_guard();
                  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
                  int descriptor = -1;
                  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
                  if (expected_signal == SIGBUS) {
                    // A mapping beyond a zero-length memfd produces a real SIGBUS.
                    descriptor = memfd_create("rocjitsu-host-fault", MFD_CLOEXEC);
                    if (descriptor < 0)
                      _exit(2);
                    flags = MAP_SHARED;
                  }
                  void *page =
                      mmap(nullptr, page_size, PROT_READ | PROT_WRITE, flags, descriptor, 0);
                  if (page == MAP_FAILED ||
                      (expected_signal == SIGSEGV && mprotect(page, page_size, PROT_NONE) != 0))
                    _exit(3);
                  (void)rocjitsu::with_host_access_guard(
                      [&] { *static_cast<volatile unsigned char *>(page) = 1; });
                  _exit(4);
                }()),
                ::testing::ExitedWithCode(64 + expected_signal), "");
  }
}

TEST(HostAccessGuardTest, DisabledRecoveryPreservesHostActions) {
  for (int signal_number : {SIGSEGV, SIGBUS}) {
    struct sigaction before {};
    ASSERT_EQ(sigaction(signal_number, nullptr, &before), 0);
    rocjitsu::install_host_access_guard();
    struct sigaction after {};
    ASSERT_EQ(sigaction(signal_number, nullptr, &after), 0);
    EXPECT_EQ(after.sa_sigaction, before.sa_sigaction);
    EXPECT_EQ(after.sa_flags, before.sa_flags);
    for (int member = 1; member < NSIG; ++member)
      EXPECT_EQ(sigismember(&after.sa_mask, member), sigismember(&before.sa_mask, member));
  }
}
#endif
