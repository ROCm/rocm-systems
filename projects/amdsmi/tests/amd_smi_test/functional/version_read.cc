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

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "version_read.h"

TestVersionRead::TestVersionRead() : TestBase() {
  set_title("AMDSMI Version Read Test");
  set_description("The Version Read tests verifies that the AMDSMI library "
                                             "version can be read properly.");
}

TestVersionRead::~TestVersionRead(void) {
}

void TestVersionRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestVersionRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestVersionRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestVersionRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestVersionRead::Run(void) {
  amdsmi_status_t err;
  amdsmi_version_t ver = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, nullptr};

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  err = amdsmi_get_lib_version(&ver);
  CHK_ERR_ASRT(err)

  ASSERT_TRUE(ver.major != 0xFFFFFFFF && ver.minor != 0xFFFFFFFF &&
              ver.release != 0xFFFFFFFF && ver.build != nullptr);
  IF_VERB(STANDARD) {
    std::cout << "\t**AMD SMI Library version: " << ver.major << "." <<
      ver.minor << "." << ver.release << " (" << ver.build << ")" << std::endl;
  }
}
