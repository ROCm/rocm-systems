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

TEST(SystemUnit, EsmiErrMsgRejectsNullOutPtr) {
  EXPECT_EQ(amdsmi_get_esmi_err_msg(AMDSMI_STATUS_SUCCESS, nullptr), AMDSMI_STATUS_INVAL);
}

}  // namespace
