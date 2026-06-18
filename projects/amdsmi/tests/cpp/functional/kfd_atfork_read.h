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

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_KFD_ATFORK_READ_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_KFD_ATFORK_READ_H_

#include "../test_base.h"

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
