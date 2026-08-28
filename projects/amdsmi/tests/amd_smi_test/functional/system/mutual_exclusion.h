// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_MUTUAL_EXCLUSION_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_MUTUAL_EXCLUSION_H_

#include <string>

#include "test_base.h"

class TestMutualExclusion : public TestBase {
 public:
  TestMutualExclusion();

  // @Brief: Destructor for test case of TestMutualExclusion
  virtual ~TestMutualExclusion();

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrieve the resource
  virtual void Close();

  // @Brief: Display  results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

 private:
  bool sleeper_process_;
  int child_;
  std::string orig_cross_process_env_;
  bool orig_cross_process_env_was_set_;
  // Pipe-based init handshake (replaces sleep-based ordering):
  //   init_pipe_:         sleeper → tester  (sleeper amdsmi_init complete)
  //   tester_ready_pipe_: tester → sleeper  (tester amdsmi_init complete)
  int init_pipe_[2];
  int tester_ready_pipe_[2];
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_MUTUAL_EXCLUSION_H_
