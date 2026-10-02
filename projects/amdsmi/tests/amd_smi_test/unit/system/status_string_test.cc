// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Null out-pointer contract for the error-string helpers: a NULL status_string
// must return AMDSMI_STATUS_INVAL instead of being dereferenced. No GPU required.

#include <gtest/gtest.h>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

namespace {

// These are plain TEST()s with no TestBase, so the VERB() macro (which reads
// TestBase::verbosity()) is unavailable; gate on the global verbosity instead.
// VERBOSE_STANDARD == 1.

TEST(SystemUnit, StatusCodeToStringRejectsNullOutPtr) {
  DISPLAY_AMDSMI_API("amdsmi_status_code_to_string", "status_string=nullptr",
                     GetTestVerbosity() >= 1);
  EXPECT_EQ(amdsmi_status_code_to_string(AMDSMI_STATUS_SUCCESS, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(SystemUnit, StatusCodeToStringValidOutPtr) {
  const char* msg = nullptr;
  DISPLAY_AMDSMI_API("amdsmi_status_code_to_string", "status_string=&msg",
                     GetTestVerbosity() >= 1);
  EXPECT_EQ(amdsmi_status_code_to_string(AMDSMI_STATUS_SUCCESS, &msg), AMDSMI_STATUS_SUCCESS);
  EXPECT_NE(msg, nullptr);
}

#if ENABLE_ESMI_LIB
TEST(SystemUnit, EsmiErrMsgRejectsNullOutPtr) {
  DISPLAY_AMDSMI_API("amdsmi_get_esmi_err_msg", "status_string=nullptr", GetTestVerbosity() >= 1);
  EXPECT_EQ(amdsmi_get_esmi_err_msg(AMDSMI_STATUS_SUCCESS, nullptr), AMDSMI_STATUS_INVAL);
}
#endif

}  // namespace
