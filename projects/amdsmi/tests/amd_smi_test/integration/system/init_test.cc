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

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kVerbose;

// amdsmi_init / amdsmi_shut_down / amdsmi_get_lib_version /
// amdsmi_status_code_to_string.

// ---- amdsmi_init + amdsmi_shut_down : refcount-balanced valid pair ----
// The shared framework already holds an init open for the duration of the
// fixture. Adding one init here bumps the refcount; the paired shut_down below
// returns it to the pre-test value, so no dangling init is left behind.
TEST_F(SystemIntegration, InitShutDown_Balanced) {
  DISPLAY_AMDSMI_API("amdsmi_init", "flags=AMDSMI_INIT_ALL_PROCESSORS", kVerbose);
  amdsmi_status_t err = amdsmi_init(AMDSMI_INIT_ALL_PROCESSORS);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  if (err == AMDSMI_STATUS_SUCCESS) {
    DISPLAY_AMDSMI_API("amdsmi_shut_down", "balance extra init", kVerbose);
    amdsmi_status_t serr = amdsmi_shut_down();
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, serr, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    AMDSMI_EXPECT_STATUS(serr, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                         AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    EXPECT_EQ(serr, AMDSMI_STATUS_SUCCESS);
  }
}

// ---- amdsmi_get_lib_version : invalid params first ----
TEST_F(SystemIntegration, GetLibVersion_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_lib_version", "version=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_lib_version(nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetLibVersion_Valid) {
  amdsmi_version_t version;
  memset(&version, 0, sizeof(version));
  DISPLAY_AMDSMI_API("amdsmi_get_lib_version", "valid out", kVerbose);
  amdsmi_status_t err = amdsmi_get_lib_version(&version);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}

// ---- amdsmi_status_code_to_string : invalid params first ----
TEST_F(SystemIntegration, StatusCodeToString_NullOutput) {
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "amdsmi_status_code_to_string crashes on a null output pointer; proper return "
         "should be AMDSMI_STATUS_INVAL";
  // Proper contract once fixed:
  //   amdsmi_status_t err = amdsmi_status_code_to_string(AMDSMI_STATUS_SUCCESS, nullptr);
  //   AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, StatusCodeToString_AllCodes) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_status_code_to_string");
  static constexpr amdsmi_status_t kCodes[] = {AMDSMI_STATUS_SUCCESS,
                                               AMDSMI_STATUS_INVAL,
                                               AMDSMI_STATUS_NO_PERM,
                                               AMDSMI_STATUS_NOT_SUPPORTED,
                                               AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                                               AMDSMI_STATUS_NOT_FOUND,
                                               AMDSMI_STATUS_NOT_INIT};
  for (auto code : kCodes) {
    const char* status_string = nullptr;
    DISPLAY_AMDSMI_API("amdsmi_status_code_to_string", "status=" + std::to_string(code), kVerbose);
    amdsmi_status_t err = amdsmi_status_code_to_string(code, &status_string);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("status=" + std::to_string(code), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}
