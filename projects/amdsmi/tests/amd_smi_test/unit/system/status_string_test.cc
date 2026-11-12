// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Null out-pointer contract for the error-string helpers: a NULL status_string
// must return AMDSMI_STATUS_INVAL instead of being dereferenced. No GPU required.

#include <gtest/gtest.h>

#include <string>

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

// MORE_DATA once had no case and fell through to "An unknown error occurred".
TEST(SystemUnit, StatusCodeToStringMoreData) {
  const char* msg = nullptr;
  ASSERT_EQ(amdsmi_status_code_to_string(AMDSMI_STATUS_MORE_DATA, &msg), AMDSMI_STATUS_SUCCESS);
  ASSERT_NE(msg, nullptr);
  EXPECT_EQ(std::string(msg).rfind("AMDSMI_STATUS_MORE_DATA:", 0), 0u) << msg;
}

#if ENABLE_ESMI_LIB
TEST(SystemUnit, EsmiErrMsgRejectsNullOutPtr) {
  EXPECT_EQ(amdsmi_get_esmi_err_msg(AMDSMI_STATUS_SUCCESS, nullptr), AMDSMI_STATUS_INVAL);
}
#endif

}  // namespace
