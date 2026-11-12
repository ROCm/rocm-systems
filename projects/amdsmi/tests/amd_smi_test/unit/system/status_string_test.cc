// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Null out-pointer contract for the error-string helpers: a NULL status_string
// must return AMDSMI_STATUS_INVAL instead of being dereferenced. No GPU required.

#include <gtest/gtest.h>

#include "amd_smi/amdsmi.h"

namespace {

TEST(SystemUnit, StatusCodeToStringRejectsNullOutPtr) {
  EXPECT_EQ(amdsmi_status_code_to_string(AMDSMI_STATUS_SUCCESS, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(SystemUnit, StatusCodeToStringValidOutPtr) {
  const char* msg = nullptr;
  EXPECT_EQ(amdsmi_status_code_to_string(AMDSMI_STATUS_SUCCESS, &msg), AMDSMI_STATUS_SUCCESS);
  EXPECT_NE(msg, nullptr);
}

#if ENABLE_ESMI_LIB
TEST(SystemUnit, EsmiErrMsgRejectsNullOutPtr) {
  EXPECT_EQ(amdsmi_get_esmi_err_msg(AMDSMI_STATUS_SUCCESS, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(SystemUnit, EsmiErrMsgLooksUpByAmdsmiStatus) {
  const char* msg = nullptr;
  EXPECT_EQ(amdsmi_get_esmi_err_msg(AMDSMI_STATUS_INVAL, &msg), AMDSMI_STATUS_SUCCESS);
  ASSERT_NE(msg, nullptr);
  EXPECT_STREQ(msg, "AMDSMI_STATUS_INVAL: Invalid parameters.");
}

TEST(SystemUnit, EsmiErrMsgUnmappedStatusIsNotSupported) {
  const char* msg = nullptr;
  EXPECT_EQ(amdsmi_get_esmi_err_msg(AMDSMI_STATUS_NOT_FOUND, &msg), AMDSMI_STATUS_NOT_SUPPORTED);
}
#endif

}  // namespace
