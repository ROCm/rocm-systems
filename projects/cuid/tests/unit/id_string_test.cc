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

#include "unit/id_string_test.h"

#include <gtest/gtest.h>

#include <cstdio>

TestIdString::TestIdString() {
  SetTitle("ID String");
  SetDescription(
      "Verify amdcuid_id_to_string formats a known ID into the expected "
      "UUID string representation.");
}

void TestIdString::SetUp() {}

void TestIdString::Run() {
  amdcuid_id_t test_id = {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8,
                          0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x0};
  const char* id_str = amdcuid_id_to_string(test_id);
  EXPECT_NE(id_str, nullptr);
  EXPECT_STREQ(id_str, "01020304-0506-0708-090a-0b0c0d0e0f00");

  IF_VERB(1) { printf("  ID string: %s\n", id_str); }
}

void TestIdString::DisplayTestInfo() { TestBase::DisplayTestInfo(); }
void TestIdString::DisplayResults() const { TestBase::DisplayResults(); }
void TestIdString::Close() {}
