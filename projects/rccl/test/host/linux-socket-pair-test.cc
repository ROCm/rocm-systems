/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/os/linux_socket_pair.cc on real pipes; only a pipe() failure is hooked.

#include <gtest/gtest.h>

#include <cerrno>
#include <fcntl.h>
#include <functional>
#include <string>
#include <unistd.h>

#include "checks.h"
#include "os_socket_pair.h"

#include "linux-test-common.h"

#include "fakes/libc_seam.h"
#include LINUX_SOCKET_PAIR_CC_PATH
#include "fakes/libc_seam_undef.h"

namespace {

bool FdOpen(int fd) { return fcntl(fd, F_GETFD) != -1; }

class LinuxSocketPairMicrotest : public LinuxTestBase {
 protected:
  void TearDown() override { ncclOsSocketPairClose(pair_); }
  ncclSocketPairDescriptor pair_[2] = {NCCL_SOCKET_PAIR_INVALID, NCCL_SOCKET_PAIR_INVALID};
};

TEST_F(LinuxSocketPairMicrotest, Create_WriteEndFeedsReadEnd) {
  ASSERT_EQ(ncclOsSocketPairCreate(pair_), ncclSuccess);
  size_t n = 0;
  ASSERT_EQ(ncclOsSocketPairWrite(pair_[1], "hello", 5, &n), ncclSuccess);
  EXPECT_EQ(n, 5u);
  char buf[8] = {};
  ASSERT_EQ(ncclOsSocketPairRead(pair_[0], buf, sizeof(buf), &n), ncclSuccess);
  EXPECT_EQ(n, 5u);
  EXPECT_STREQ(buf, "hello");
}

TEST_F(LinuxSocketPairMicrotest, Read_ZeroBytesAfterTheWriterCloses) {
  ASSERT_EQ(ncclOsSocketPairCreate(pair_), ncclSuccess);
  ::close(pair_[1]);
  pair_[1] = NCCL_SOCKET_PAIR_INVALID;
  char buf[4];
  size_t n = 99;
  ASSERT_EQ(ncclOsSocketPairRead(pair_[0], buf, sizeof(buf), &n), ncclSuccess);
  EXPECT_EQ(n, 0u);
}

TEST_F(LinuxSocketPairMicrotest, Close_ClosesBothEndsAndSkipsInvalidOnes) {
  ASSERT_EQ(ncclOsSocketPairCreate(pair_), ncclSuccess);
  int fds[2] = {pair_[0], pair_[1]};
  ASSERT_EQ(ncclOsSocketPairClose(pair_), ncclSuccess);
  EXPECT_FALSE(FdOpen(fds[0]));
  EXPECT_FALSE(FdOpen(fds[1]));
  EXPECT_EQ(pair_[0], NCCL_SOCKET_PAIR_INVALID);
  EXPECT_EQ(pair_[1], NCCL_SOCKET_PAIR_INVALID);
  EXPECT_EQ(ncclOsSocketPairClose(pair_), ncclSuccess);

  ASSERT_EQ(ncclOsSocketPairCreate(pair_), ncclSuccess);
  int readEnd = pair_[0];
  ::close(pair_[1]);
  pair_[1] = NCCL_SOCKET_PAIR_INVALID;
  ASSERT_EQ(ncclOsSocketPairClose(pair_), ncclSuccess);
  EXPECT_FALSE(FdOpen(readEnd));
  EXPECT_TRUE(log_.Text().empty());
}

// Both ends are still reset, and only the first failure is reported.
TEST_F(LinuxSocketPairMicrotest, Close_ReportsTheFirstFailure) {
  ASSERT_EQ(ncclOsSocketPairCreate(pair_), ncclSuccess);
  ::close(pair_[0]);
  ::close(pair_[1]);
  EXPECT_EQ(ncclOsSocketPairClose(pair_), ncclSystemError);
  EXPECT_EQ(pair_[0], NCCL_SOCKET_PAIR_INVALID);
  EXPECT_EQ(pair_[1], NCCL_SOCKET_PAIR_INVALID);
  EXPECT_EQ(log_.Text(), "WARN close failed: Bad file descriptor\n");

  ASSERT_EQ(ncclOsSocketPairCreate(pair_), ncclSuccess);
  int readEnd = pair_[0];
  ::close(pair_[1]);
  EXPECT_EQ(ncclOsSocketPairClose(pair_), ncclSystemError);  // the second end fails
  EXPECT_FALSE(FdOpen(readEnd));
}

TEST_F(LinuxSocketPairMicrotest, Create_PipeFailureIsASystemError) {
  ScopedHook pipeHook(g_pipe, [](int*) {
    errno = EMFILE;
    return -1;
  });
  EXPECT_EQ(ncclOsSocketPairCreate(pair_), ncclSystemError);
  EXPECT_EQ(pair_[0], NCCL_SOCKET_PAIR_INVALID);
  EXPECT_TRUE(log_.Has("WARN Call to pipe failed: Too many open files"));
}

TEST_F(LinuxSocketPairMicrotest, ReadAndWrite_FailuresAreSystemErrors) {
  ASSERT_EQ(ncclOsSocketPairCreate(pair_), ncclSuccess);
  size_t n = 99;
  char buf[4];
  EXPECT_EQ(ncclOsSocketPairWrite(pair_[0], "x", 1, &n), ncclSystemError);  // the read end
  EXPECT_TRUE(log_.Has("WARN Call to write failed"));
  EXPECT_EQ(ncclOsSocketPairRead(pair_[1], buf, sizeof(buf), &n), ncclSystemError);
  EXPECT_TRUE(log_.Has("WARN Call to read failed"));
  EXPECT_EQ(n, 99u);
}

}  // namespace
