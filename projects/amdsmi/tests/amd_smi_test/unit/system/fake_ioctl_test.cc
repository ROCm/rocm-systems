// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Checks ScopedFakeIoctl itself and shows how to use it. The request codes are arbitrary.

#include <gtest/gtest.h>
#include <sys/ioctl.h>

#include <cerrno>

#include "test_fixture_utils.h"

namespace {

using amd::smi::testing::ScopedFakeIoctl;

constexpr unsigned long kDouble = 0x5a01;
constexpr unsigned long kNegate = 0x5a02;
constexpr unsigned long kUnknown = 0x5a03;

struct DoubleArgs {
  int in;
  int out;
};

TEST(SystemUnit, FAKE_IOCTL__HandlerFillsTypedArgsAndIsCounted) {
  ScopedFakeIoctl fake;
  fake.RegisterHandler<DoubleArgs>(kDouble, [](auto& args) {
    args.out = args.in * 2;
    return 0;
  });

  DoubleArgs args = {21, 0};
  EXPECT_EQ(ioctl(-1, kDouble, &args), 0);
  EXPECT_EQ(args.out, 42);
  EXPECT_EQ(fake.CallCount(kDouble), 1);
}

TEST(SystemUnit, FAKE_IOCTL__FailureReturnsMinusOneWithErrno) {
  ScopedFakeIoctl fake;
  fake.RegisterFailure(kDouble, EACCES);

  EXPECT_EQ(ioctl(-1, kDouble, nullptr), -1);
  EXPECT_EQ(errno, EACCES);
}

TEST(SystemUnit, FAKE_IOCTL__UnhandledRequestFailsWithENOTTY) {
  ScopedFakeIoctl fake;
  fake.RegisterHandler(kDouble, [](void*) { return 0; });

  EXPECT_EQ(ioctl(-1, kUnknown, nullptr), -1);
  EXPECT_EQ(errno, ENOTTY);
  EXPECT_EQ(fake.CallCount(kUnknown), 0);
}

TEST(SystemUnit, FAKE_IOCTL__NestedFakePassesUnhandledRequestsToOuter) {
  ScopedFakeIoctl outer;
  outer.RegisterHandler(kDouble, [](void*) { return 1; });
  {
    ScopedFakeIoctl inner;
    inner.RegisterHandler(kNegate, [](void*) { return 2; });

    EXPECT_EQ(ioctl(-1, kNegate, nullptr), 2);  // the inner fake answers
    EXPECT_EQ(ioctl(-1, kDouble, nullptr), 1);  // passed on to the outer fake
    EXPECT_EQ(ioctl(-1, kUnknown, nullptr), -1);
    EXPECT_EQ(errno, ENOTTY);
  }
  EXPECT_EQ(ioctl(-1, kNegate, nullptr), -1);  // the inner fake is gone
  EXPECT_EQ(errno, ENOTTY);
  EXPECT_EQ(ioctl(-1, kDouble, nullptr), 1);
}

TEST(SystemUnit, FAKE_IOCTL__NoFakeActiveCallsTheRealIoctl) {
  // fd -1 is invalid, so the kernel answers EBADF; a fake would have answered ENOTTY.
  EXPECT_EQ(ioctl(-1, kDouble, nullptr), -1);
  EXPECT_EQ(errno, EBADF);
}

}  // namespace
