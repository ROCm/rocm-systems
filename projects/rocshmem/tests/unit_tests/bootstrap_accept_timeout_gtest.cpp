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

/* Regression test: Socket::accept() must honour its timeout after a peer has
 * been discarded.
 *
 * A connection that fails the magic check is rejected by discardAccept(), which
 * returns the socket to SocketStateAccepting. Before the poll gate in
 * Socket::accept(), the loop then re-entered a BLOCKING ::accept() and the
 * timeout check below it was never evaluated, so the listener waited forever
 * for a replacement peer. Measured on the unfixed code: a 5s timeout still
 * blocked at 60s.
 *
 * The blocking-accept defect is older than the hang it now causes. #12853
 * correctly made the wrong-magic path non-fatal (it previously called
 * ERROR() -> abort(), which killed rank 0 on a stray port scan), but the retry
 * it enabled re-enters a blocking accept(). Before that commit the retry was
 * dead code, so the unreachable timeout never showed.
 *
 * The listener runs in a forked child because a timed-out accept() calls
 * ERROR() == LOG_ERROR_ABORT, i.e. abort(). The property under test is that the
 * child TERMINATES within a bounded window; by which route it dies does not
 * matter, and hanging is the failure. The parent does the waiting, so this is
 * an ordinary TEST_F rather than a gtest death test -- EXPECT_EXIT would wait
 * on the child indefinitely and hang the suite instead of failing it.
 *
 * Needs no GPU, no MPI ranks and no network beyond loopback.
 */

#include <cstring>

#include "bootstrap_accept_timeout_gtest.hpp"

namespace rocshmem {

TEST_F(BootstrapAcceptTimeoutFixture, AcceptTimesOutAfterPeerDiscarded) {
  const uint16_t port = reserveLoopbackPort();
  ASSERT_NE(port, 0) << "could not reserve a loopback port";

  // Readiness pipe: the listener tells us when it is actually bound. See
  // waitForListenerReady() for why this is not a fixed settle delay.
  int ready[2] = {-1, -1};
  ASSERT_EQ(::pipe(ready), 0) << "pipe failed: " << strerror(errno);

  pid_t listener = fork();
  ASSERT_NE(listener, -1) << "fork failed for listener";
  if (listener == 0) {
    ::close(ready[0]);
    SocketAddress addr = loopbackAddr(port);
    Socket l(&addr, ROCSHMEM_SOCKET_MAGIC, SocketTypeBootstrap);
    l.bindAndListen();
    if (l.getState() != SocketStateReady) {
      const char f = 'F';
      ssize_t ignored = ::write(ready[1], &f, 1);
      (void)ignored;
      _exit(90);
    }
    // Bound and listening: the port is now accepting connections, so a connect
    // from here on reaches finalizeAccept() rather than being refused.
    const char r = 'R';
    ssize_t ignored = ::write(ready[1], &r, 1);
    (void)ignored;
    ::close(ready[1]);
    Socket peer;
    // NB: Socket timeouts are MICROSECONDS, not seconds.
    peer.accept(&l, kAcceptTimeoutUs);
    _exit(peer.getState() == SocketStateReady ? 0 : 1);
  }
  ::close(ready[1]);

  const char token = waitForListenerReady(ready[0], kListenerReadyLimitSeconds);
  ::close(ready[0]);
  if (token != 'R') {
    ::kill(listener, SIGKILL);
    waitpid(listener, nullptr, 0);
  }
  // Hard failure, not a skip: without a bound listener the prober would be
  // refused, nothing would be discarded, and a green result would be meaningless.
  ASSERT_EQ(token, 'R') << "listener never reported ready (token="
                        << static_cast<int>(token)
                        << "); it did not bind, so the discard path would not "
                           "have been exercised";

  pid_t prober = fork();
  ASSERT_NE(prober, -1) << "fork failed for prober";
  if (prober == 0) {
    SocketAddress addr = loopbackAddr(port);
    // Wrong magic: finalizeAccept() rejects this and calls discardAccept(),
    // which is the state the listener used to get stuck in.
    Socket c(&addr, kWrongMagic, SocketTypeBootstrap);
    c.connect(kAcceptTimeoutUs);
    _exit(0);
  }

  waitpid(prober, nullptr, 0);

  // Poll rather than block in waitpid(), so a hang is REPORTED as a failure
  // instead of hanging the whole unit-test binary with it.
  const double elapsed = waitForExit(listener, kMaxWallSeconds);

  EXPECT_LT(elapsed, kMaxWallSeconds)
      << "listener was still in accept() after " << elapsed << "s for a "
      << (kAcceptTimeoutUs / 1e6)
      << "s timeout -- it hung; see the regression note at the top of this file";

  // Guard against the opposite false pass: a listener that returns immediately
  // never waited on the discarded peer at all. The timeout is the lower bound.
  // Half of it, not all of it: the listener starts its timer slightly before
  // the parent starts measuring, so a correct run lands just under the nominal
  // 5s. Half is far below any correct run and far above an instant return, so
  // the margin is not tight enough to flake.
  EXPECT_GE(elapsed, (kAcceptTimeoutUs / 1e6) * 0.5)
      << "listener returned after only " << elapsed
      << "s, well under its " << (kAcceptTimeoutUs / 1e6)
      << "s timeout -- it likely never entered the discard-and-retry path, so "
         "this run did not test the regression";
}

}  // namespace rocshmem
