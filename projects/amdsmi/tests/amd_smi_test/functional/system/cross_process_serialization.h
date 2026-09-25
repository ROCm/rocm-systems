// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_CROSS_PROCESS_SERIALIZATION_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_CROSS_PROCESS_SERIALIZATION_H_

#include <string>

#include "test_base.h"

// Verifies that AMDSMI_MUTEX_CROSS_PROCESS=1 (blocking mode) actually
// serializes API calls across processes. Unlike TestMutualExclusion (which
// uses RSMI_INIT_FLAG_RESRV_TEST1 to get observable BUSY returns), this test
// uses normal blocking mode and measures elapsed time to prove Process B
// waited behind Process A's held mutex rather than racing through.
class TestCrossProcessSerialization : public TestBase {
 public:
  TestCrossProcessSerialization();
  virtual ~TestCrossProcessSerialization();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

 private:
  bool holder_process_;
  bool is_child_process_;
  int child_;
  std::string orig_cross_process_env_;
  bool orig_cross_process_env_was_set_;
  // Phase-1 init sync (holder → waiter): holder writes after amdsmi_init;
  // waiter reads before calling amdsmi_init, so holder always inits first.
  int init_pipe_[2];
  // Phase-2 init sync (waiter → holder): waiter writes after amdsmi_init;
  // holder reads before entering Run(), so holder only acquires the mutex
  // after both processes have completed amdsmi_init. Without this, the
  // waiter's amdsmi_init blocks on the shared device mutex held by
  // rsmi_test_sleep, so amdsmi_get_gpu_id never sees a contended mutex.
  int waiter_ready_pipe_[2];
  // Run() ordering (holder → waiter): holder writes just before rsmi_test_sleep;
  // waiter reads before amdsmi_get_gpu_id, guaranteeing holder holds the
  // mutex before waiter attempts to acquire it.
  int run_pipe_[2];
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_CROSS_PROCESS_SERIALIZATION_H_
