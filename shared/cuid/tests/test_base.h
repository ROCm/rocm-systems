// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_BASE_H_
#define CUID_TEST_BASE_H_

#include <string>
#include <vector>

#include "include/amd_cuid.h"
#include "test_common.h"

class TestBase {
 public:
  TestBase();
  virtual ~TestBase() = default;

  // Enumerate devices and populate device_handles_. Sets setup_failed_ on
  // error so Run() can skip gracefully.
  virtual void SetUp();

  // Core test body. Subclasses must implement this.
  virtual void Run() = 0;

  // No-op for CUID (no explicit library shutdown), but available for
  // subclasses that need post-test cleanup.
  virtual void Close();

  virtual void DisplayTestInfo();
  virtual void DisplayResults() const;

  void SetTitle(const std::string& title) { title_ = title; }
  void SetDescription(const std::string& desc) { description_ = desc; }
  bool SetupFailed() const { return setup_failed_; }

 protected:
  std::string title_;
  std::string description_;

  bool setup_failed_ = false;
  std::vector<amdcuid_id_t> device_handles_;
};

// Runs the full test lifecycle: DisplayTestInfo → SetUp → Run → DisplayResults
// → Close. If SetUp sets setup_failed_, Run() is skipped.
void RunGenericTest(TestBase* test);

#endif  // CUID_TEST_BASE_H_
