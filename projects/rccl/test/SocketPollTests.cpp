/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Functional regression test for the bootstrap socket busy-loop fix.
 *
 * The NCCL_SOCKET_POLL_TIMEOUT_MSEC parameter, when set to a non-zero value,
 * makes socketWait() block in poll() until the socket becomes readable/writable
 * instead of spinning on socketProgress(), which previously pinned a CPU core at
 * ~100% for the whole duration of a bootstrap transfer that was waiting on a
 * slow / late peer.
 *
 * This test exercises the public ncclSocket* send/recv path (which routes
 * through socketWait) over a loopback connection where the sender is
 * deliberately slow. It measures the CPU time consumed by the receiving thread
 * while it is blocked waiting for data:
 *
 *   - With NCCL_SOCKET_POLL_TIMEOUT_MSEC > 0 (the fix, default on >= v2.29) the
 *     receiver sleeps in poll(), so it consumes almost no CPU relative to the
 *     wall-clock time it spends waiting.
 *   - With NCCL_SOCKET_POLL_TIMEOUT_MSEC == 0 (pre-fix behavior, and the code
 *     path taken by every release before v2.29) the receiver busy-spins, so it
 *     consumes CPU time roughly equal to the wall-clock wait.
 *
 * The "poll enabled -> low CPU" assertion passes on v2.29+ and fails on any
 * build lacking the fix (where the parameter is ignored and the loop always
 * spins), giving a clean pass-after / fail-before signal. The "poll disabled ->
 * high CPU" case is kept as a control that documents the busy-loop behavior the
 * fix removes.
 */

#include "socket.h"

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "gtest/gtest.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <netinet/in.h>
#include <thread>

namespace RcclUnitTesting {
namespace {

// Wall-clock time the sender stays idle before delivering the payload. The
// receiver is blocked inside socketWait() for (approximately) this long, which
// is the window in which busy-looping vs. polling is distinguishable.
constexpr int kSenderIdleMs = 1000;

// Payload is intentionally tiny so a single TCP segment / read completes the
// transfer once the data is finally sent.
constexpr int kPayloadBytes = 4096;

// Fraction of the wait spent burning CPU. A polling receiver stays far below
// this; a busy-looping receiver sits near 1.0. 0.5 leaves a wide margin for
// scheduler noise on loaded CI machines.
constexpr double kCpuBusyThreshold = 0.5;

double timespecDiffMs(const struct timespec &start, const struct timespec &end) {
  return (end.tv_sec - start.tv_sec) * 1e3 +
         (end.tv_nsec - start.tv_nsec) / 1e6;
}

// Loopback (127.0.0.1) address with an ephemeral port for ncclSocketListen().
union ncclSocketAddress LoopbackAddr() {
  union ncclSocketAddress addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin.sin_family = AF_INET;
  addr.sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin.sin_port = 0; // kernel assigns a free port
  return addr;
}

// Outcome of a single slow-transfer measurement, run entirely inside an
// isolated child process so NCCL_SOCKET_POLL_TIMEOUT_MSEC is read fresh.
struct TransferMeasurement {
  bool connected = false;   // connection pair was established
  bool dataValid = false;   // payload received intact
  double wallMs = 0.0;      // wall time the receiver spent in ncclSocketRecv
  double cpuMs = 0.0;       // CPU time the receiver thread burned in that span
};

// Establishes a loopback ncclSocket pair, then transfers a small payload with
// the sender deliberately delayed by kSenderIdleMs, measuring how much CPU the
// receiving thread burns while blocked in ncclSocketRecv().
TransferMeasurement RunSlowTransfer() {
  TransferMeasurement m;

  const uint64_t magic = NCCL_SOCKET_MAGIC;
  const enum ncclSocketType type = ncclSocketTypeBootstrap;

  struct ncclSocket listenSock;
  union ncclSocketAddress listenAddr = LoopbackAddr();
  if (ncclSocketInit(&listenSock, &listenAddr, magic, type) != ncclSuccess) {
    TEST_INFO("ncclSocketInit(listen) failed");
    return m;
  }
  if (ncclSocketListen(&listenSock) != ncclSuccess) {
    TEST_INFO("ncclSocketListen failed");
    return m;
  }

  union ncclSocketAddress boundAddr;
  if (ncclSocketGetAddr(&listenSock, &boundAddr) != ncclSuccess) {
    TEST_INFO("ncclSocketGetAddr failed");
    ncclSocketClose(&listenSock);
    return m;
  }

  // Accept side runs on its own thread; the connect side drives the main
  // thread. Both handshakes (magic/type exchange) must progress concurrently.
  struct ncclSocket acceptSock;
  memset(&acceptSock, 0, sizeof(acceptSock));
  std::atomic<bool> acceptOk{false};
  std::thread acceptThread([&]() {
    if (ncclSocketInit(&acceptSock, nullptr, magic, type) != ncclSuccess) {
      TEST_INFO("ncclSocketInit(accept) failed");
      return;
    }
    if (ncclSocketAccept(&acceptSock, &listenSock) != ncclSuccess) {
      TEST_INFO("ncclSocketAccept failed");
      return;
    }
    acceptOk.store(acceptSock.state == ncclSocketStateReady);
  });

  struct ncclSocket connectSock;
  union ncclSocketAddress connectAddr = boundAddr;
  bool connectOk = false;
  if (ncclSocketInit(&connectSock, &connectAddr, magic, type) == ncclSuccess &&
      ncclSocketConnect(&connectSock) == ncclSuccess) {
    connectOk = (connectSock.state == ncclSocketStateReady);
  } else {
    TEST_INFO("ncclSocketConnect failed");
  }

  acceptThread.join();

  if (!connectOk || !acceptOk.load()) {
    TEST_INFO("Failed to establish socket pair (connect=%d accept=%d)",
              connectOk ? 1 : 0, acceptOk.load() ? 1 : 0);
    ncclSocketClose(&connectSock);
    ncclSocketClose(&acceptSock);
    ncclSocketClose(&listenSock);
    return m;
  }
  m.connected = true;

  // Receiver: connect side. Blocks in ncclSocketRecv() -> socketWait() until
  // the (delayed) payload arrives, measuring its own thread CPU consumption.
  std::vector<char> recvBuf(kPayloadBytes, 0);
  std::thread recvThread([&]() {
    struct timespec cpu0, cpu1, wall0, wall1;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu0);
    clock_gettime(CLOCK_MONOTONIC, &wall0);

    ncclResult_t r = ncclSocketRecv(&connectSock, recvBuf.data(), kPayloadBytes);

    clock_gettime(CLOCK_MONOTONIC, &wall1);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu1);

    m.wallMs = timespecDiffMs(wall0, wall1);
    m.cpuMs = timespecDiffMs(cpu0, cpu1);
    m.dataValid = (r == ncclSuccess);
  });

  // Sender: accept side. Stay idle so the receiver is parked in socketWait,
  // then deliver the whole payload at once.
  std::vector<char> sendBuf(kPayloadBytes);
  for (int i = 0; i < kPayloadBytes; ++i)
    sendBuf[i] = static_cast<char>(i & 0xff);
  std::this_thread::sleep_for(std::chrono::milliseconds(kSenderIdleMs));
  ncclResult_t sendRes =
      ncclSocketSend(&acceptSock, sendBuf.data(), kPayloadBytes);

  recvThread.join();

  if (sendRes == ncclSuccess && m.dataValid)
    m.dataValid = (memcmp(sendBuf.data(), recvBuf.data(), kPayloadBytes) == 0);
  else
    m.dataValid = false;

  ncclSocketClose(&connectSock);
  ncclSocketClose(&acceptSock);
  ncclSocketClose(&listenSock);
  return m;
}

// Shared body: run the slow transfer and check correctness + the CPU ratio.
// expectBusy selects the assertion direction (control vs. fix).
void CheckSlowTransfer(bool expectBusy) {
  TransferMeasurement m = RunSlowTransfer();

  if (!m.connected) {
    GTEST_SKIP() << "Could not establish a loopback socket pair; "
                 << "networking may be unavailable in this environment.";
    return;
  }

  ASSERT_TRUE(m.dataValid) << "Payload was not received intact over the socket pair.";
  ASSERT_GT(m.wallMs, kSenderIdleMs * 0.5)
      << "Receiver returned too quickly (wall=" << m.wallMs
      << "ms); the slow-sender wait was not exercised.";

  const double ratio = m.cpuMs / m.wallMs;
  TEST_INFO("Slow transfer: wall=%.1fms cpu=%.1fms ratio=%.3f (expectBusy=%d)",
            m.wallMs, m.cpuMs, ratio, expectBusy ? 1 : 0);

  if (expectBusy) {
    // Control: NCCL_SOCKET_POLL_TIMEOUT_MSEC=0 reproduces the pre-fix busy loop.
    EXPECT_GT(ratio, kCpuBusyThreshold)
        << "Expected the receiver to busy-spin when polling is disabled, but it "
        << "consumed little CPU (cpu=" << m.cpuMs << "ms / wall=" << m.wallMs
        << "ms). ratio=" << ratio;
  } else {
    // Fix: with polling enabled the receiver must sleep, not spin.
    EXPECT_LT(ratio, kCpuBusyThreshold)
        << "Receiver busy-spun while waiting for a slow peer even though "
        << "NCCL_SOCKET_POLL_TIMEOUT_MSEC is set. This indicates the bootstrap "
        << "socket poll fix is missing or ineffective. cpu="
        << m.cpuMs << "ms / wall=" << m.wallMs << "ms, ratio=" << ratio;
  }
}

} // namespace

// The bug fix: NCCL_SOCKET_POLL_TIMEOUT_MSEC > 0 makes a blocked bootstrap
// receive wait in poll() instead of burning a CPU core. Passes on v2.29+
// (fix present) and fails on pre-fix builds where the parameter is ignored.
TEST(SocketPollTimeout, PollEnabledDoesNotBusyLoop) {
  ProcessIsolatedTestRunner::ExecutionOptions options;
  options.stopOnFirstFailure = false;
  options.verboseLogging = true;

  RUN_ISOLATED_TESTS_WITH_OPTIONS(
      options,
      ProcessIsolatedTestRunner::TestConfig(
          "PollEnabledDoesNotBusyLoop",
          []() { CheckSlowTransfer(/*expectBusy=*/false); })
          .withEnvironment({{"NCCL_SOCKET_POLL_TIMEOUT_MSEC", "5000"}})
          .withTimeout(std::chrono::seconds(60)));
}

// Control: NCCL_SOCKET_POLL_TIMEOUT_MSEC=0 selects the exact code path used by
// every pre-v2.29 build, demonstrating the busy loop the fix removes.
TEST(SocketPollTimeout, PollDisabledBusyLoops) {
  ProcessIsolatedTestRunner::ExecutionOptions options;
  options.stopOnFirstFailure = false;
  options.verboseLogging = true;

  RUN_ISOLATED_TESTS_WITH_OPTIONS(
      options,
      ProcessIsolatedTestRunner::TestConfig(
          "PollDisabledBusyLoops",
          []() { CheckSlowTransfer(/*expectBusy=*/true); })
          .withEnvironment({{"NCCL_SOCKET_POLL_TIMEOUT_MSEC", "0"}})
          .withTimeout(std::chrono::seconds(60)));
}

} // namespace RcclUnitTesting
