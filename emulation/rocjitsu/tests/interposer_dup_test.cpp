// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interposer_dup_test.cpp
/// @brief LD_PRELOAD regression tests for the interposer's KFD fd/backend
///        bookkeeping across dup / dup2 / dup3 / fcntl(F_DUPFD).
///
/// @details These run with librocjitsu.so preloaded (see the ENVIRONMENT set in
/// tests/CMakeLists.txt) so that open("/dev/kfd") is serviced by the simulated
/// KFD driver. AMDKFD_IOC_GET_VERSION is used purely as a routing probe: it
/// succeeds (returns 0 and fills the version) only when the fd is routed to a
/// KFD backend, and fails when the fd falls through to the real (non-KFD)
/// descriptor. The SimulatedDriver-level unit tests (KfdIoctlTest) cannot catch
/// these because they exercise the driver object directly, bypassing the fd
/// tracking that lives in the interposer.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>

namespace {

// Issue AMDKFD_IOC_GET_VERSION on fd. Returns true if the fd routed to a KFD
// backend (ioctl succeeded and reported the expected major version).
bool kfd_version_ok(int fd) {
  kfd_ioctl_get_version_args args{};
  int rc = ioctl(fd, AMDKFD_IOC_GET_VERSION, &args);
  return rc == 0 && args.major_version == KFD_IOCTL_MAJOR_VERSION;
}

int open_kfd() { return open("/dev/kfd", O_RDWR | O_CLOEXEC); }

} // namespace

// A plain dup() of the KFD fd must keep routing KFD ioctls to the driver, and
// closing the original primary fd must not tear the process down while the dup
// still holds a reference (dup keeps the backend alive).
TEST(InterposerDupTest, DupKeepsKfdRoutingAfterPrimaryClose) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  EXPECT_TRUE(kfd_version_ok(kfd));

  int dup_fd = dup(kfd);
  ASSERT_GE(dup_fd, 0);
  EXPECT_TRUE(kfd_version_ok(dup_fd));

  // Close the primary; the dup must still route to the live KFD backend.
  EXPECT_EQ(close(kfd), 0);
  EXPECT_TRUE(kfd_version_ok(dup_fd));

  EXPECT_EQ(close(dup_fd), 0);
}

// fcntl(F_DUPFD_CLOEXEC) is the dup path libdrm uses; it must also preserve KFD
// routing on the duplicate.
TEST(InterposerDupTest, FcntlDupfdKeepsKfdRouting) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);

  int dup_fd = fcntl(kfd, F_DUPFD_CLOEXEC, 0);
  ASSERT_GE(dup_fd, 0);
  EXPECT_TRUE(kfd_version_ok(dup_fd));

  EXPECT_EQ(close(dup_fd), 0);
  // Original still routes.
  EXPECT_TRUE(kfd_version_ok(kfd));
  EXPECT_EQ(close(kfd), 0);
}

// dup2 that OVERWRITES the primary KFD fd number must invalidate the old primary
// identity: after dup2(other, kfd) the kfd number now names 'other', so KFD
// ioctls on it must NOT be routed to the (now-replaced) KFD backend, and closing
// it must behave like closing a normal fd.
TEST(InterposerDupTest, Dup2OverPrimaryInvalidatesKfdIdentity) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  EXPECT_TRUE(kfd_version_ok(kfd));

  // A plain pipe fd to overwrite the primary KFD fd number with.
  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);

  // Overwrite the KFD primary number with the read end of the pipe.
  ASSERT_EQ(dup2(pipefd[0], kfd), kfd);

  // The kfd number now refers to the pipe, not the KFD backend: a KFD ioctl must
  // no longer succeed against it.
  EXPECT_FALSE(kfd_version_ok(kfd));

  // Cleanup. Closing the overwritten number closes the pipe read-end dup.
  EXPECT_EQ(close(kfd), 0);
  EXPECT_EQ(close(pipefd[0]), 0);
  EXPECT_EQ(close(pipefd[1]), 0);
}

// dup2 of the KFD fd ONTO a fresh number must make the target route KFD ioctls,
// and the reference bookkeeping must let both fds be closed without prematurely
// destroying the backend.
TEST(InterposerDupTest, Dup2OntoFreshFdRoutesKfd) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);

  // Reserve a target fd number with a pipe end, then dup2 the KFD fd onto it.
  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);
  int target = pipefd[0];

  ASSERT_EQ(dup2(kfd, target), target);
  EXPECT_TRUE(kfd_version_ok(target));

  // Close the primary; the dup2 target keeps the backend alive.
  EXPECT_EQ(close(kfd), 0);
  EXPECT_TRUE(kfd_version_ok(target));

  EXPECT_EQ(close(target), 0);
  EXPECT_EQ(close(pipefd[1]), 0);
}

// dup3 of the KFD fd onto a fresh number must route KFD ioctls to the target,
// and dup3(fd, fd, flags) must fail with EINVAL without disturbing tracking (the
// interposer's reserve/reconcile path must roll back cleanly on that failure).
TEST(InterposerDupTest, Dup3RoutesKfdAndRejectsSameFd) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  EXPECT_TRUE(kfd_version_ok(kfd));

  // dup3(fd, fd, ...) is required to fail with EINVAL and leave fd untouched.
  errno = 0;
  EXPECT_EQ(dup3(kfd, kfd, O_CLOEXEC), -1);
  EXPECT_EQ(errno, EINVAL);
  // The primary must still route after the rejected dup3 (no tracking disturbed).
  EXPECT_TRUE(kfd_version_ok(kfd));

  // dup3 onto a fresh number routes KFD, and both fds close cleanly.
  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);
  int target = pipefd[0];
  ASSERT_EQ(dup3(kfd, target, O_CLOEXEC), target);
  EXPECT_TRUE(kfd_version_ok(target));

  EXPECT_EQ(close(kfd), 0);
  EXPECT_TRUE(kfd_version_ok(target)); // dup3 target keeps the backend alive.

  EXPECT_EQ(close(target), 0);
  EXPECT_EQ(close(pipefd[1]), 0);
}

// Overwriting the primary KFD fd number via dup2 while a dup keeps the backend
// alive must (a) leave the dup routing, and (b) let a fresh open("/dev/kfd")
// return a valid, routable KFD fd. This is the reopen-after-overwrite path that
// re-mints the primary fd number (remote: reissue_synthetic_kfd_fd under
// remote_mutex_; local: ensure_fd_created under process_mutex_) without
// disturbing the still-live backend the dup holds. Runs identically on the local
// and daemon (remote) harnesses.
TEST(InterposerDupTest, ReopenAfterPrimaryOverwriteKeepsBackend) {
  int kfd = open_kfd();
  ASSERT_GE(kfd, 0);
  EXPECT_TRUE(kfd_version_ok(kfd));

  // A dup keeps the backend alive across the primary's overwrite.
  int keeper = dup(kfd);
  ASSERT_GE(keeper, 0);
  EXPECT_TRUE(kfd_version_ok(keeper));

  // Overwrite the primary fd number with an unrelated pipe end. After dup2 the
  // kfd number aliases the pipe read-end, so pipefd[0] is redundant and must be
  // closed to avoid leaking a descriptor across the other tests in this process.
  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);
  ASSERT_EQ(dup2(pipefd[0], kfd), kfd);
  EXPECT_EQ(close(pipefd[0]), 0);
  // The overwritten number now names the pipe, not the KFD backend.
  EXPECT_FALSE(kfd_version_ok(kfd));
  // The dup still routes: the backend stayed alive.
  EXPECT_TRUE(kfd_version_ok(keeper));

  // A fresh open must return a valid, routable KFD fd (re-minted primary).
  int kfd2 = open_kfd();
  ASSERT_GE(kfd2, 0) << "reopen after primary overwrite returned " << kfd2;
  EXPECT_TRUE(kfd_version_ok(kfd2));

  EXPECT_EQ(close(kfd2), 0);
  EXPECT_EQ(close(keeper), 0);
  EXPECT_EQ(close(kfd), 0); // closes the pipe-read dup installed over the number
  EXPECT_EQ(close(pipefd[1]), 0);
}

// Serialized reopen-after-overwrite under contention. The interposer keeps a
// single primary KFD fd slot, so a distinct dup (keeper) holds the backend alive
// while the main thread repeatedly overwrites the primary fd number and reopens
// "/dev/kfd" — always exactly one primary at a time, matching how a real client
// uses /dev/kfd. A background thread churns dup/close on the keeper so backend
// retain/release runs concurrently with invalidation + reopen. With invalidation
// and open serialized on the same lock (remote_mutex_ / process_mutex_), every
// reopen must return a routable KFD fd — never -1 or a reused non-KFD descriptor
// (the ENOTTY case the review reproduced). This is the invariant asserted here,
// and it holds on both the local (process_mutex_) and daemon/remote
// (remote_mutex_) harnesses. Note: the keeper's own routability is deliberately
// not asserted after the loop — on the local backend a reopen rebinds the shared
// backing fd and untracks existing dups (clear_dups), which is expected
// local-only behavior unrelated to the reopen-routability invariant under test.
TEST(InterposerDupTest, SerializedReopenUnderContentionStaysRoutable) {
  int primary = open_kfd();
  ASSERT_GE(primary, 0);
  ASSERT_TRUE(kfd_version_ok(primary));
  int keeper = dup(primary); // distinct number; holds the backend across reopens
  ASSERT_GE(keeper, 0);
  ASSERT_TRUE(kfd_version_ok(keeper));

  constexpr int kIters = 500;
  std::atomic<bool> stop{false};

  // Background churn: dup the keeper and close it, exercising backend
  // retain/release concurrently with the reopen loop below. A short yield/sleep
  // between iterations keeps the contention window open without spinning a full
  // core (which would add CI flakiness/timeouts under parallel test runs).
  std::thread churn([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      int d = dup(keeper);
      if (d >= 0)
        close(d);
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  });

  // Use non-fatal checks (EXPECT_*/break) rather than ASSERT_* inside the loop:
  // the churn thread is joinable here, so a fatal assertion that returned early
  // would skip stop/join and terminate the process. Any setup failure breaks to
  // the shutdown path below.
  int bad = 0;
  bool setup_failed = false;
  for (int i = 0; i < kIters; ++i) {
    // Overwrite the current primary fd number with a pipe end, releasing the
    // primary's reference (keeper keeps the backend alive). This drives
    // invalidate_overwritten_kfd_fd() concurrently with the churn thread. After
    // dup2 the primary number aliases the pipe read-end, so close both the
    // now-redundant pipefd[0] and the aliased primary number, plus pipefd[1].
    int pipefd[2];
    if (pipe(pipefd) != 0) {
      setup_failed = true;
      break;
    }
    if (dup2(pipefd[0], primary) != primary) {
      setup_failed = true;
      close(pipefd[0]);
      close(pipefd[1]);
      break;
    }
    close(primary); // now the pipe-read dup installed over the number
    close(pipefd[0]);
    close(pipefd[1]);

    // Reopen: the slot was cleared, so this must re-mint a fresh, routable
    // primary (a distinct number from the still-open keeper).
    primary = open_kfd();
    if (primary < 0) {
      ++bad;
      primary = dup(keeper); // recover so the loop can continue overwriting
      if (primary < 0) {     // recovery also failed under fd pressure: stop.
        setup_failed = true;
        break;
      }
      continue;
    }
    if (!kfd_version_ok(primary))
      ++bad;
  }

  stop.store(true);
  churn.join();

  EXPECT_FALSE(setup_failed) << "pipe()/dup2() failed under resource pressure";
  EXPECT_EQ(bad, 0) << "a reopen returned -1 or a non-routable fd under contention";
  if (primary >= 0) {
    EXPECT_EQ(close(primary), 0);
  }
  EXPECT_EQ(close(keeper), 0);
}
