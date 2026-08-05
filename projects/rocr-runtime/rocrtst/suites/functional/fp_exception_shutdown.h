/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ROCRTST_SUITES_FUNCTIONAL_FP_EXCEPTION_SHUTDOWN_H_
#define ROCRTST_SUITES_FUNCTIONAL_FP_EXCEPTION_SHUTDOWN_H_

#include <sys/types.h>

#include "common/base_rocr.h"
#include "suites/test_common/test_base.h"

// Regression test for a SIGFPE raised inside libhsa-runtime64 during
// hsa_shut_down() when the host process has FP exception trapping enabled
// (fesetenv(FE_NOMASK_ENV) + fedisableexcept(FE_UNDERFLOW | FE_INEXACT),
// leaving FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID active).
//
// Runtime-side fix: core/inc/signal.h::GetFastTimeout guards hsa_freq == 0
// (rocm-systems PR #5148).
//
// The init/queue-create/shutdown sequence runs in a forked child so that
// the parent's HSA state and gtest harness are not affected by the strict
// FP environment, and a SIGFPE in the runtime is observable as a child
// termination signal rather than killing the test process.
class FpExceptionShutdownTest : public TestBase {
 public:
  FpExceptionShutdownTest(void);
  virtual ~FpExceptionShutdownTest(void);

  // SetUp/Close are overridden to avoid hsa_init() / hsa_shut_down() in
  // the parent; the runtime is only initialized inside the forked child.
  virtual void SetUp(void);
  virtual void Close(void);

  virtual void Run(void);
  virtual void DisplayTestInfo(void);
  virtual void DisplayResults(void) const;

  void TestShutdownSurvivesStrictFpEnv(void);

 private:
  // Fork a child that enables strict FP exception trapping, initializes
  // HSA, creates a queue, and calls hsa_shut_down. Returns the child PID,
  // or -1 on fork() failure / unsupported platform. The caller must
  // waitpid() the returned PID.
  pid_t RunShutdownInChild(void);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_FP_EXCEPTION_SHUTDOWN_H_
