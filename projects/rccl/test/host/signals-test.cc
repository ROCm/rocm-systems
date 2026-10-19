/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/signals.cc: the opt-in handler registration (signal() is hooked)
// and the handler itself, whose exit path runs in a death test.

#include <gtest/gtest.h>

#include <csignal>
#include <cstdio>
#include <execinfo.h>
#include <functional>
#include <string>
#include <unistd.h>
#include <vector>

#include "../common/LogCapture.hpp"
#include "debug.h"
#include "param.h"

#include "linux-test-common.h"

#undef HAVE_BFD  // the libbfd backtrace needs libbfd, which this binary does not link
#include "fakes/param_redirect.h"
#include "fakes/libc_seam.h"
#include SIGNALS_CC_PATH
#include "fakes/libc_seam_undef.h"

namespace {

using RcclUnitTesting::CaptureStdout;

const std::vector<int> kCaught = {SIGILL, SIGBUS, SIGFPE, SIGSEGV, SIGUSR2};

class SignalsMicrotest : public LinuxTestBase {
 protected:
  void SetUp() override {
    for (int sig : kCaught) sigaction(sig, nullptr, &saved_[sig]);
  }
  void TearDown() override {
    for (int sig : kCaught) sigaction(sig, &saved_[sig], nullptr);
  }
  static sighandler_t Installed(int sig) {
    struct sigaction sa;
    sigaction(sig, nullptr, &sa);
    return sa.sa_handler;
  }
  struct sigaction saved_[NSIG];
};

TEST_F(SignalsMicrotest, Register_OffByDefault) {
  ScopedHook signalHook(g_signal, [](int, sighandler_t) { return SIG_DFL; });
  RegisterSignalHandlers();
  EXPECT_EQ(signalHook.calls, 0);
  EXPECT_TRUE(log_.Text().empty());
}

TEST_F(SignalsMicrotest, Register_InstallsTheHandlerForTheFaultSignalsAndUsr2) {
  params_.Set("RCCL_ENABLE_SIGNALHANDLER", 1);
  std::vector<int> seen;
  ScopedHook signalHook(g_signal, [&](int sig, sighandler_t h) {
    seen.push_back(sig);
    EXPECT_EQ(h, &sig_handler);
    return ::signal(sig, h);
  });
  RegisterSignalHandlers();
  EXPECT_EQ(seen, kCaught);
  for (int sig : kCaught) EXPECT_EQ(Installed(sig), &sig_handler) << sig;
  EXPECT_TRUE(log_.Has("INFO Enabling custom signal handler"));
  EXPECT_FALSE(log_.Has("Unable to register"));
}

TEST_F(SignalsMicrotest, Register_FailureIsOnlyLogged) {
  params_.Set("RCCL_ENABLE_SIGNALHANDLER", 1);
  ScopedHook signalHook(g_signal, [](int, sighandler_t) { return SIG_ERR; });
  RegisterSignalHandlers();
  EXPECT_TRUE(log_.Has("Unable to register signal handler for Illegal instruction"));
  EXPECT_TRUE(log_.Has("Unable to register signal handler for User defined signal 2"));
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
  params_.Set("RCCL_ENABLE_SIGNALHANDLER", 1);
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
