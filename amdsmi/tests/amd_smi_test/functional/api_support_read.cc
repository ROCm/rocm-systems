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


#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include "api_support_read.h"

TestAPISupportRead::TestAPISupportRead() : TestBase() {
  set_title("AMDSMI API Support Read Test");
  set_description("This test verifies that the supported APIs are correctly "
                                                               "identified.");
}

TestAPISupportRead::~TestAPISupportRead(void) {
}

void TestAPISupportRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestAPISupportRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestAPISupportRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestAPISupportRead::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestAPISupportRead::Run(void) {
  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) {
      std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    }
    return;
  }

}
