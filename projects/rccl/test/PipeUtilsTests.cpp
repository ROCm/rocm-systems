/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "common/PipeUtils.hpp"

namespace RcclUnitTesting
{
  namespace
  {
    struct Pipe
    {
      int fds[2] = {-1, -1};
      Pipe() { EXPECT_EQ(pipe(fds), 0); }
      ~Pipe() { CloseRead(); CloseWrite(); }
      int  ReadFd()  const { return fds[0]; }
      int  WriteFd() const { return fds[1]; }
      void CloseRead()  { if (fds[0] >= 0) { close(fds[0]); fds[0] = -1; } }
      void CloseWrite() { if (fds[1] >= 0) { close(fds[1]); fds[1] = -1; } }
    };
  }

  TEST(PipeUtils, RoundTripFullCount)
  {
    Pipe p;
    uint64_t const sent = 0x0123456789abcdefULL;
    ASSERT_EQ(detail::safe_pipe_write(p.WriteFd(), &sent, sizeof(sent)),
              static_cast<ssize_t>(sizeof(sent)));
    uint64_t received = 0;
    ASSERT_EQ(detail::safe_pipe_read(p.ReadFd(), &received, sizeof(received)),
              static_cast<ssize_t>(sizeof(received)));
    EXPECT_EQ(received, sent);
  }

  // A payload larger than the pipe buffer forces partial writes and reads.
  TEST(PipeUtils, RoundTripLargerThanPipeBuffer)
  {
    Pipe p;
    int const pipeSize = fcntl(p.WriteFd(), F_GETPIPE_SZ);
    ASSERT_GT(pipeSize, 0);
    std::vector<uint8_t> sent(static_cast<size_t>(pipeSize) * 4);
    for (size_t i = 0; i < sent.size(); ++i) sent[i] = static_cast<uint8_t>(i * 31);

    pid_t const writer = fork();
    ASSERT_GE(writer, 0);
    if (writer == 0)
    {
      p.CloseRead();
      ssize_t const written = detail::safe_pipe_write(p.WriteFd(), sent.data(), sent.size());
      _exit(written == static_cast<ssize_t>(sent.size()) ? 0 : 1);
    }
    p.CloseWrite();

    std::vector<uint8_t> received(sent.size());
    EXPECT_EQ(detail::safe_pipe_read(p.ReadFd(), received.data(), received.size()),
              static_cast<ssize_t>(received.size()));
    EXPECT_EQ(received, sent);

    int writerStatus = 0;
    ASSERT_EQ(waitpid(writer, &writerStatus, 0), writer);
    EXPECT_TRUE(WIFEXITED(writerStatus) && WEXITSTATUS(writerStatus) == 0);
  }

  TEST(PipeUtils, ReadReturnsZeroOnImmediateEof)
  {
    Pipe p;
    p.CloseWrite();
    uint64_t received = 0;
    EXPECT_EQ(detail::safe_pipe_read(p.ReadFd(), &received, sizeof(received)), 0);
  }

  TEST(PipeUtils, ReadReturnsShortCountOnEof)
  {
    Pipe p;
    uint8_t const sent[3] = {1, 2, 3};
    ASSERT_EQ(detail::safe_pipe_write(p.WriteFd(), sent, sizeof(sent)),
              static_cast<ssize_t>(sizeof(sent)));
    p.CloseWrite();

    uint8_t received[8] = {};
    EXPECT_EQ(detail::safe_pipe_read(p.ReadFd(), received, sizeof(received)),
              static_cast<ssize_t>(sizeof(sent)));
    EXPECT_EQ(std::memcmp(received, sent, sizeof(sent)), 0);
  }

  TEST(PipeUtils, ReadAndWriteFailOnBadFd)
  {
    uint64_t value = 0;
    errno = 0;
    EXPECT_EQ(detail::safe_pipe_read(-1, &value, sizeof(value)), -1);
    EXPECT_EQ(errno, EBADF);
    errno = 0;
    EXPECT_EQ(detail::safe_pipe_write(-1, &value, sizeof(value)), -1);
    EXPECT_EQ(errno, EBADF);
  }
}
