// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interposer_errno_test.cpp
/// @brief Verify the interposer translates driver ioctl results into the libc
/// ioctl(2) return/errno contract.

#include "rocjitsu/kmd/linux/kfd_ioctl_result.h"

#include <gtest/gtest.h>

#include <cerrno>

namespace {

using rocjitsu::kmd::kfd_ioctl_ret;

TEST(InterposerErrno, SuccessZeroPassesThrough) {
  errno = 0;
  EXPECT_EQ(kfd_ioctl_ret(0), 0);
  EXPECT_EQ(errno, 0);
}

TEST(InterposerErrno, PositiveResultPassesThrough) {
  // DBG_TRAP SUSPEND_QUEUES/RESUME_QUEUES return the number of queues handled,
  // a meaningful positive value that must not be treated as an error.
  errno = 0;
  EXPECT_EQ(kfd_ioctl_ret(3), 3);
  EXPECT_EQ(errno, 0);
}

TEST(InterposerErrno, NegativeErrnoBecomesMinusOneWithErrno) {
  // The stale-errno bug that aborted rocgdb's runtime handshake: -EAGAIN from
  // QUERY_DEBUG_EVENT must surface as return -1 with errno == EAGAIN, not a raw
  // negative that a caller misreads against a stale errno.
  errno = 0;
  EXPECT_EQ(kfd_ioctl_ret(-EAGAIN), -1);
  EXPECT_EQ(errno, EAGAIN);
}

TEST(InterposerErrno, EachErrnoIsPreserved) {
  for (int e : {EINVAL, EPERM, ESRCH, ENODEV, ENOMEM}) {
    errno = 0;
    EXPECT_EQ(kfd_ioctl_ret(-e), -1);
    EXPECT_EQ(errno, e);
  }
}

} // namespace
