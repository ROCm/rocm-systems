// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "version_read.h"

#include <gtest/gtest.h>

#include <iostream>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestVersionRead::TestVersionRead() : TestBase() {
  set_title("AMDSMI Version Read Test");
  set_description(
      "The Version Read tests verifies that the AMDSMI library "
      "version can be read properly.");
}

TestVersionRead::~TestVersionRead(void) {}

void TestVersionRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestVersionRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

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
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  DISPLAY_AMDSMI_API("amdsmi_get_lib_version", "", VERB(STANDARD));
  err = amdsmi_get_lib_version(&ver);
  DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  CHK_ERR_ASRT(err)

  ASSERT_TRUE(ver.major != 0xFFFFFFFF && ver.minor != 0xFFFFFFFF && ver.release != 0xFFFFFFFF &&
              ver.build != nullptr);
  IF_VERB(STANDARD) {
    std::cout << "\t**AMD SMI Library version: " << ver.major << "." << ver.minor << "."
              << ver.release << " (" << ver.build << ")" << std::endl;
  }
}
