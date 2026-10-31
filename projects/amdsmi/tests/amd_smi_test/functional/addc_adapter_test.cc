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

// Unit tests for the ADDC error-summary JSON -> AFID adapter. The parser is
// toolchain-independent and builds on the default GCC-11 path (no ENABLE_ADDC).

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_addc.h"

namespace {

TEST(AddcAdapter, ParsesAfidsInOrder) {
  std::vector<uint64_t> out;
  ASSERT_TRUE(amdsmi::addc::parse_afids_from_summary(
      R"([{"fru":"AID0","afid":12345},{"fru":"AID1","afid":67890}])", out));
  EXPECT_EQ(out, (std::vector<uint64_t>{12345, 67890}));
}

TEST(AddcAdapter, EmptyArrayYieldsNoAfids) {
  std::vector<uint64_t> out{99};
  ASSERT_TRUE(amdsmi::addc::parse_afids_from_summary("[]", out));
  EXPECT_TRUE(out.empty());
}

TEST(AddcAdapter, SkipsEntriesWithoutAfid) {
  std::vector<uint64_t> out;
  ASSERT_TRUE(
      amdsmi::addc::parse_afids_from_summary(R"([{"fru":"AID0"},{"fru":"AID1","afid":42}])", out));
  EXPECT_EQ(out, (std::vector<uint64_t>{42}));
}

TEST(AddcAdapter, MalformedJsonFailsAndClears) {
  std::vector<uint64_t> out{7};
  EXPECT_FALSE(amdsmi::addc::parse_afids_from_summary("not json", out));
  EXPECT_TRUE(out.empty());
}

TEST(AddcAdapter, ObjectRootFailsAndClears) {
  std::vector<uint64_t> out{5};
  EXPECT_FALSE(amdsmi::addc::parse_afids_from_summary(R"({"afid":1})", out));
  EXPECT_TRUE(out.empty());
}

TEST(AddcAdapter, SkipsNonUnsignedAfidValues) {
  std::vector<uint64_t> out;
  ASSERT_TRUE(amdsmi::addc::parse_afids_from_summary(
      R"([{"afid":-1},{"afid":"x"},{"afid":1.5},{"afid":7}])", out));
  EXPECT_EQ(out, (std::vector<uint64_t>{7}));
}

TEST(AddcAdapter, SkipsNonObjectArrayEntries) {
  std::vector<uint64_t> out;
  ASSERT_TRUE(amdsmi::addc::parse_afids_from_summary(R"([123,{"afid":1}])", out));
  EXPECT_EQ(out, (std::vector<uint64_t>{1}));
}

}  // namespace
