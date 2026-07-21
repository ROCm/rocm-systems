/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

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
