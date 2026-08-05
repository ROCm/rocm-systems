/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// _GNU_SOURCE must be defined before any system header so that the glibc
// extensions FE_NOMASK_ENV and fedisableexcept() are exposed by <cfenv>.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cfenv>
#include <csignal>
#include <cstdio>
#include <cstring>

#include "suites/functional/fp_exception_shutdown.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

// Child-to-parent exit codes. 0/1 mirror the gpu_coredump.cc convention;
// 2 is added so the child can ask the parent to record a skip when a
// runtime precondition (e.g., visible GPU) is not met.
static const int kChildExitPass = 0;
static const int kChildExitFail = 1;
static const int kChildExitSkip = 2;

// Bound waitpid() so a hang in hsa_shut_down (a plausible failure mode
// for InterruptSignal teardown bugs) cannot stall the suite.
static const int kChildTimeoutMs = 10000;
static const int kChildPollMs = 10;

FpExceptionShutdownTest::FpExceptionShutdownTest(void) : TestBase() {
  // Hard-coded to 1: each iteration forks and creates a GPU queue, which
  // is expensive; the test is not designed for repetition.
  set_num_iteration(1);
  set_title("FP Exception Shutdown Test");
  set_description(
      "Regression test for SIGFPE in hsa_shut_down() under strict FP "
      "exception trapping (FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID). "
      "Runtime-side fix: core/inc/signal.h::GetFastTimeout guards "
      "hsa_freq == 0 (rocm-systems PR #5148).");
}

FpExceptionShutdownTest::~FpExceptionShutdownTest(void) {
}

void FpExceptionShutdownTest::SetUp(void) {
  if (!checkPlatformFiltering()) return;

  // TestBase::SetUp() intentionally skipped: it would hsa_init() in the
  // parent, but all HSA work for this test must run inside the forked child.
  //
  // Defensive check: if a prior test in this gtest process left HSA
  // initialized (reference count not fully drained by hsa_shut_down()),
  // fork() would inherit live runtime threads in the child and behavior
  // would be undefined. hsa_shut_down() returns NOT_INITIALIZED iff no
  // init is outstanding; anything else means another test polluted global
  // state. Fail the test and skip the unsafe fork rather than producing a
  // flaky result.
  hsa_status_t shutdown_probe = hsa_shut_down();
  if (shutdown_probe != HSA_STATUS_ERROR_NOT_INITIALIZED) {
    ADD_FAILURE() << "Another rocrtst test left HSA initialized in this "
                     "process (hsa_shut_down returned "
                  << shutdown_probe
                  << "). Forking with live HSA threads is undefined "
                     "behavior; verify test ordering and that prior tests "
                     "fully tear down HSA state.";
    test_skipped_ = true;
    return;
  }

#ifndef FE_NOMASK_ENV
  fprintf(stdout, "[ SKIPPED ] FE_NOMASK_ENV not available on this platform\n");
  test_skipped_ = true;
#endif
}

void FpExceptionShutdownTest::Run(void) {
  // TestBase::Run() intentionally skipped: this test does not emit the
  // standard execution banner because the work happens via fork() in the
  // test method itself.
}

void FpExceptionShutdownTest::Close(void) {
  // TestBase::Close() intentionally skipped: no parent-side HSA state to
  // tear down (see SetUp).
}

void FpExceptionShutdownTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void FpExceptionShutdownTest::DisplayResults(void) const {
  // No bespoke results; results banner omitted to match this test's overall
  // minimal output (no Run/Close banners either).
}

pid_t FpExceptionShutdownTest::RunShutdownInChild(void) {
#ifndef FE_NOMASK_ENV
  // Unreachable in practice: SetUp() sets test_skipped_ on platforms
  // without this macro, so the test method never runs. Stub kept so the
  // file compiles on non-glibc targets.
  return -1;
#else
  pid_t pid = fork();
  if (pid < 0) {
    return -1;
  }

  if (pid == 0) {
    // Enable strict FP exception trapping in the child only.
    if (fesetenv(FE_NOMASK_ENV) != 0) {
      fprintf(stderr, "CHILD: fesetenv(FE_NOMASK_ENV) failed\n");
      _exit(kChildExitFail);
    }
    fedisableexcept(FE_UNDERFLOW | FE_INEXACT);
    // Active exceptions: FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID

    hsa_status_t err = hsa_init();
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: hsa_init failed (%d)\n", err);
      _exit(kChildExitFail);
    }

    hsa_agent_t gpu_agent = {0};
    err = hsa_iterate_agents(rocrtst::FindGPUDevice, &gpu_agent);
    // FindGPUDevice returns HSA_STATUS_INFO_BREAK on success.
    if (err == HSA_STATUS_INFO_BREAK) {
      err = HSA_STATUS_SUCCESS;
    } else if (err == HSA_STATUS_SUCCESS) {
      err = HSA_STATUS_ERROR;
    }
    if (err != HSA_STATUS_SUCCESS || gpu_agent.handle == 0) {
      // No GPU visible (no device, ROCR_VISIBLE_DEVICES filtering, etc.).
      // Ask the parent to record a skip rather than a failure.
      fprintf(stderr, "CHILD: no GPU agent visible; skipping\n");
      hsa_shut_down();
      _exit(kChildExitSkip);
    }

    uint32_t queue_size = 0;
    err = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                             &queue_size);
    if (err != HSA_STATUS_SUCCESS || queue_size == 0) {
      fprintf(stderr, "CHILD: failed to query queue size\n");
      hsa_shut_down();
      _exit(kChildExitFail);
    }

    // Create the AqlQueue whose destructor is the crash site in the original
    // bug. Leave it live so hsa_shut_down() drives the teardown path that
    // matches the failing stack trace: Runtime::Unload -> GpuAgent::
    // ReleaseResources -> AqlQueue::~AqlQueue -> InterruptSignal::WaitRelaxed.
    hsa_queue_t* queue = nullptr;
    err = hsa_queue_create(gpu_agent, queue_size, HSA_QUEUE_TYPE_MULTI, nullptr,
                           nullptr, 0, 0, &queue);
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: hsa_queue_create failed (%d)\n", err);
      hsa_shut_down();
      _exit(kChildExitFail);
    }

    // Drive shutdown explicitly so any SIGFPE is attributable to this exact
    // call, rather than to a later finalizer-triggered call we couldn't
    // pinpoint in the stack trace.
    err = hsa_shut_down();
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: hsa_shut_down returned %d\n", err);
      _exit(kChildExitFail);
    }

    _exit(kChildExitPass);
  }

  return pid;
#endif  // FE_NOMASK_ENV
}

void FpExceptionShutdownTest::TestShutdownSurvivesStrictFpEnv(void) {
  pid_t child = RunShutdownInChild();
  ASSERT_GT(child, 0) << "fork() failed: " << strerror(errno);

  int status = 0;
  int elapsed_ms = 0;
  pid_t reaped = 0;
  while (elapsed_ms < kChildTimeoutMs) {
    reaped = waitpid(child, &status, WNOHANG);
    if (reaped == child) break;
    // waitpid() can return -1 with EINTR if interrupted by a signal; that is
    // not a real failure, so keep polling. Only bail out on other errors.
    if (reaped < 0 && errno != EINTR) {
      FAIL() << "waitpid() failed: " << strerror(errno);
    }
    usleep(kChildPollMs * 1000);
    elapsed_ms += kChildPollMs;
  }

  if (reaped != child) {
    kill(child, SIGKILL);
    waitpid(child, &status, 0);
    FAIL() << "Child timed out after " << kChildTimeoutMs
           << " ms (possible hang in hsa_shut_down).";
  }

  if (WIFEXITED(status)) {
    int code = WEXITSTATUS(status);
    if (code == kChildExitSkip) {
      // NOTE: the gtest vendored under rocrtst/gtest/ predates GTEST_SKIP(),
      // so we cannot record a first-class "skipped" outcome. This test will
      // appear as PASSED in the gtest summary even when the precondition
      // was not met. Grep CI logs for the "[ SKIPPED ]" marker below to
      // distinguish a true pass from a skipped run.
      fprintf(stdout, "[ SKIPPED ] precondition not met in child; see CHILD: "
                      "stderr above\n");
      return;
    }
    EXPECT_EQ(kChildExitPass, code)
        << "Child exited with status " << code
        << "; see CHILD: stderr output above for the failing step.";
    return;
  }

  if (WIFSIGNALED(status)) {
    int sig = WTERMSIG(status);
    if (sig == SIGFPE) {
      FAIL() << "Regression: hsa_shut_down() raised SIGFPE under strict FP "
                "exception trapping.";
    }
    FAIL() << "Child terminated by unexpected signal " << sig << " ("
           << strsignal(sig) << ")";
  }

  FAIL() << "Child neither exited nor was signalled (raw status=" << status
         << ")";
}
