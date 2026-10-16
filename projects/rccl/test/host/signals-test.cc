/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/signals.cc: the opt-in handler registration (signal() is seamed)
// and the handler itself, whose exit path runs in a death test.

#include <gtest/gtest.h>

#include <csignal>
#include <cstdio>
#include <functional>
#include <string>
#include <unistd.h>
#include <vector>

#include "../common/LogCapture.hpp"
#include "debug.h"
#include "param.h"

#include "linux-test-common.h"

namespace {
std::function<sighandler_t(int, sighandler_t)> g_signal = ::signal;
}

#undef HAVE_BFD  // the libbfd backtrace needs libbfd, which this binary does not link
#define signal(...) g_signal(__VA_ARGS__)
#include SIGNALS_CC_PATH
#undef signal

namespace {

using RcclUnitTesting::CaptureStdout;

const std::vector<int> kCaught = {SIGILL, SIGBUS, SIGFPE, SIGSEGV, SIGUSR2};

class SignalsMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_linuxLog.clear();
    for (int sig : kCaught) sigaction(sig, nullptr, &saved_[sig]);
  }
  void TearDown() override {
    for (int sig : kCaught) sigaction(sig, &saved_[sig], nullptr);
    g_signal = ::signal;
  }
  static sighandler_t Installed(int sig) {
    struct sigaction sa;
    sigaction(sig, nullptr, &sa);
    return sa.sa_handler;
  }
  struct sigaction saved_[NSIG];
};

TEST_F(SignalsMicrotest, Register_OffByDefault) {
  LinuxScopedEnv env("RCCL_ENABLE_SIGNALHANDLER", nullptr);
  int calls = 0;
  g_signal = [&](int, sighandler_t) {
    calls++;
    return SIG_DFL;
  };
  RegisterSignalHandlers();
  EXPECT_EQ(calls, 0);
  EXPECT_TRUE(g_linuxLog.empty());
}

TEST_F(SignalsMicrotest, Register_InstallsTheHandlerForTheFaultSignalsAndUsr2) {
  LinuxScopedEnv env("RCCL_ENABLE_SIGNALHANDLER", "1");
  std::vector<int> seen;
  g_signal = [&](int sig, sighandler_t h) {
    seen.push_back(sig);
    EXPECT_EQ(h, &sig_handler);
    return ::signal(sig, h);
  };
  RegisterSignalHandlers();
  EXPECT_EQ(seen, kCaught);
  for (int sig : kCaught) EXPECT_EQ(Installed(sig), &sig_handler) << sig;
  EXPECT_TRUE(LinuxLogged("INFO Enabling custom signal handler"));
  EXPECT_FALSE(LinuxLogged("Unable to register"));
}

TEST_F(SignalsMicrotest, Register_FailureIsOnlyLogged) {
  LinuxScopedEnv env("RCCL_ENABLE_SIGNALHANDLER", "1");
  g_signal = [](int, sighandler_t) { return SIG_ERR; };
  RegisterSignalHandlers();
  EXPECT_TRUE(LinuxLogged("Unable to register signal handler for Illegal instruction"));
  EXPECT_TRUE(LinuxLogged("Unable to register signal handler for User defined signal 2"));
}

TEST_F(SignalsMicrotest, Handler_Usr2PrintsABacktraceAndReturns) {
  std::string out = CaptureStdout([] {
    sig_handler(SIGUSR2);
    fflush(stdout);
  });
  std::string head = "\n[Process: " + std::to_string(getpid()) + "] Inside handler function signal: " +
                     strsignal(SIGUSR2) + " (" + std::to_string(SIGUSR2) + ")\n";
  ASSERT_EQ(out.rfind(head, 0), 0u) << out;
  EXPECT_GT(out.size(), head.size());  // the backtrace frames follow
}

TEST_F(SignalsMicrotest, Handler_RaisedUsr2ReturnsToTheProgram) {
  LinuxScopedEnv env("RCCL_ENABLE_SIGNALHANDLER", "1");
  RegisterSignalHandlers();
  std::string out = CaptureStdout([] {
    raise(SIGUSR2);
    fflush(stdout);
  });
  EXPECT_NE(out.find("Inside handler function signal"), std::string::npos);
}

TEST_F(SignalsMicrotest, Handler_FaultSignalsExitWithMinusOne) {
  for (int sig : {SIGILL, SIGBUS, SIGFPE, SIGSEGV}) {
    EXPECT_EXIT(sig_handler(sig), ::testing::ExitedWithCode(255), "") << sig;
  }
}

}  // namespace
