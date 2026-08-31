// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_UNIT_FILE_LOCK_TEST_H_
#define CUID_TEST_UNIT_FILE_LOCK_TEST_H_

#include "test_base.h"

// Each sub-test in this file is a separate class because the fork-based tests
// use blocking waits that are not safe to compose in a single Run() body.

class TestFileLockBasic : public TestBase {
 public:
  TestFileLockBasic();
  void SetUp() override;
  void Run() override;
};

class TestFileLockRAII : public TestBase {
 public:
  TestFileLockRAII();
  void SetUp() override;
  void Run() override;
};

class TestFileLockMultipleShared : public TestBase {
 public:
  TestFileLockMultipleShared();
  void SetUp() override;
  void Run() override;
};

class TestFileLockExclusiveBlocks : public TestBase {
 public:
  TestFileLockExclusiveBlocks();
  void SetUp() override;
  void Run() override;
};

class TestFileLockTimeout : public TestBase {
 public:
  TestFileLockTimeout();
  void SetUp() override;
  void Run() override;
};

class TestFileLockTimeoutSpecialCases : public TestBase {
 public:
  TestFileLockTimeoutSpecialCases();
  void SetUp() override;
  void Run() override;
};

#endif  // CUID_TEST_UNIT_FILE_LOCK_TEST_H_
