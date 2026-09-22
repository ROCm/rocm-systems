// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_KFD_ATFORK_READ_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_KFD_ATFORK_READ_H_

#include "test_base.h"

// Verifies that querying GPU memory usage does not trigger a caller's
// pthread_atfork handlers. The KFD VRAM helper spawns a short-lived child via
// clone() rather than fork() specifically to bypass glibc's atfork dispatch,
// which can recurse/deadlock when AMD-SMI is called from an atfork chain
// (ROCM-24163). A regression to fork() would fire the handlers below.
class TestKfdAtforkRead : public TestBase {
 public:
  TestKfdAtforkRead();

  virtual ~TestKfdAtforkRead();

  virtual void SetUp();

  virtual void Run();

  virtual void Close();

  virtual void DisplayResults() const;

  virtual void DisplayTestInfo(void);
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_KFD_ATFORK_READ_H_
