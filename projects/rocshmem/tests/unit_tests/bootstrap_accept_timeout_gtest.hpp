/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef ROCSHMEM_BOOTSTRAP_ACCEPT_TIMEOUT_GTEST_HPP
#define ROCSHMEM_BOOTSTRAP_ACCEPT_TIMEOUT_GTEST_HPP

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>

#include "gtest/gtest.h"
#include "../src/bootstrap/socket.hpp"

namespace rocshmem {

class BootstrapAcceptTimeoutFixture : public ::testing::Test {
 public:
  // The listener is given this long to notice nobody valid is coming.
  static constexpr int64_t kAcceptTimeoutUs = 5LL * 1000 * 1000;
  // Generous allowance over the timeout; the bug overruns it without bound.
  // 4x the accept timeout: large enough that scheduling noise on a loaded CI
  // box cannot reach it, small enough to report the hang quickly. The pre-fix
  // failure mode has no upper bound, so the exact value is not load-bearing --
  // an unfixed listener blows through any ceiling.
  static constexpr double kMaxWallSeconds = 20.0;
  // Upper bound on how long we will wait for the listener to report readiness.
  // This is a deadlock guard, NOT a settle time: the test proceeds as soon as
  // the pipe says the listener is up, so a slow machine costs nothing. Only a
  // listener that never binds at all reaches this, and that is a hard failure,
  // never a silent skip.
  static constexpr double kListenerReadyLimitSeconds = 30.0;

  static constexpr uint64_t kWrongMagic = 0xBADBADBADBADBADBULL;

  // Bind an ephemeral loopback port, read it back, release it. The small race
  // against another process claiming it in the gap is acceptable here and keeps
  // the test free of a hardcoded port.
  static uint16_t reserveLoopbackPort() {
    int s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return 0;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    if (::bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
      ::close(s);
      return 0;
    }
    socklen_t len = sizeof(a);
    if (::getsockname(s, reinterpret_cast<sockaddr*>(&a), &len) != 0) {
      ::close(s);
      return 0;
    }
    uint16_t port = ntohs(a.sin_port);
    ::close(s);
    return port;
  }

  static SocketAddress loopbackAddr(uint16_t port) {
    SocketAddress addr{};
    addr.sin.sin_family = AF_INET;
    addr.sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin.sin_port = htons(port);
    return addr;
  }

  // Wait up to limitSeconds for pid to terminate; returns elapsed seconds.
  // Reaps -- killing first if needed -- either way, so a hung listener cannot
  // outlive the test holding its port, and no zombie is left behind.
  static double waitForExit(pid_t pid, double limitSeconds) {
    const auto start = std::chrono::steady_clock::now();
    auto since = [&start]() {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                           start)
          .count();
    };
    for (;;) {
      int status = 0;
      if (waitpid(pid, &status, WNOHANG) == pid) return since();
      if (since() > limitSeconds) break;
      usleep(100000);
    }
    const double elapsed = since();
    ::kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    return elapsed;
  }

  // Block until the listener child writes its readiness byte, or the guard
  // expires. Returns the byte: 'R' = bound and in accept(), 'F' = the child
  // failed to bind, 0 = nothing arrived (child died early, or never got there).
  //
  // Deliberately NOT a fixed sleep. A blind settle delay fails toward a FALSE
  // PASS: if the listener has not bound yet, the prober's connect is refused,
  // no peer is ever discarded, and accept() simply times out on its own -- the
  // test goes green having never exercised discardAccept(), the whole point.
  // Waiting on an explicit signal makes that case impossible.
  static char waitForListenerReady(int fd, double limitSeconds) {
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
      char tok = 0;
      const ssize_t n = ::read(fd, &tok, 1);
      if (n == 1) return tok;
      if (n == 0) return 0;  // write end closed: child exited without signalling
      if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
        return 0;
      }
      const double waited =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count();
      if (waited > limitSeconds) return 0;
      usleep(1000);
    }
  }
};

}  // namespace rocshmem

#endif  // ROCSHMEM_BOOTSTRAP_ACCEPT_TIMEOUT_GTEST_HPP
