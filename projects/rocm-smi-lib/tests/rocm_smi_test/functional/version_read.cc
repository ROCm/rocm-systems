/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <stdint.h>
#include <stddef.h>

#include <iostream>
#include <map>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/version_read.h"
#include "rocm_smi_test/test_common.h"

TestVersionRead::TestVersionRead() : TestBase() {
  set_title("RSMI Version Read Test");
  set_description("The Version Read tests verifies that the RSMI library "
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
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const uint32_t kVerMaxStrLen = 80;

static const std::map<uint32_t, const char *>
  kComponentNameMap = {
      {RSMI_SW_COMP_DRIVER, "Driver Version"},
};

void TestVersionRead::Run(void) {
  rsmi_status_t err;
  rsmi_version_t ver = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, nullptr};

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  err = rsmi_version_get(&ver);
  CHK_ERR_ASRT(err)

  ASSERT_TRUE(ver.major != 0xFFFFFFFF && ver.minor != 0xFFFFFFFF &&
                             ver.patch != 0xFFFFFFFF && ver.build != nullptr);
  IF_VERB(STANDARD) {
    std::cout << "\t**RocM SMI Library version: " << ver.major << "." <<
       ver.minor << "." << ver.patch << " (" << ver.build << ")" << std::endl;
  }

  char ver_str[kVerMaxStrLen];

  for (uint32_t cmp = RSMI_SW_COMP_FIRST; cmp <= RSMI_SW_COMP_LAST; ++cmp) {
    err = rsmi_version_str_get(static_cast<rsmi_sw_component_t>(cmp),
                                                      ver_str, kVerMaxStrLen);
    CHK_ERR_ASRT(err)

    IF_VERB(STANDARD) {
      std::cout << "\t**" << kComponentNameMap.at(cmp) << ": " <<
                                                         ver_str << std::endl;
    }
  }
}
