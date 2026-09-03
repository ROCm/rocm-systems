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

// Regression coverage for the AMPP config/writable_slot_mask hex bitmask
// parser.
//
// config/writable_slot_mask is driver-owned sysfs content, but must not be
// trusted blindly. This test exercises the pure-parsing logic directly --
// no sysfs, root, or mock-sysfs overlay required -- to prove every
// malformed/adversarial input is rejected promptly with
// RSMI_STATUS_UNEXPECTED_DATA instead of silently accepting garbage.
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi_ampp.h"

namespace {

using amd::smi::parse_ampp_writable_slot_mask;

TEST(GpuUnit, AmppWritableSlotMaskParseValidMaskExpandsToBits) {
  std::vector<uint32_t> slots;
  ASSERT_EQ(parse_ampp_writable_slot_mask("0xe0", &slots), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(slots, (std::vector<uint32_t>{5, 6, 7}));
}

TEST(GpuUnit, AmppWritableSlotMaskParseUppercasePrefixAccepted) {
  std::vector<uint32_t> slots;
  ASSERT_EQ(parse_ampp_writable_slot_mask("0X1", &slots), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(slots, (std::vector<uint32_t>{0}));
}

TEST(GpuUnit, AmppWritableSlotMaskParseSingleBitMask) {
  std::vector<uint32_t> slots;
  ASSERT_EQ(parse_ampp_writable_slot_mask("0x1", &slots), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(slots, (std::vector<uint32_t>{0}));
}

TEST(GpuUnit, AmppWritableSlotMaskParseFullMaskAllSlotsWritable) {
  std::vector<uint32_t> slots;
  ASSERT_EQ(parse_ampp_writable_slot_mask("0xff", &slots), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(slots, (std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(GpuUnit, AmppWritableSlotMaskParseEmptyLineMeansNothingWritable) {
  std::vector<uint32_t> slots{1, 2, 3};  // pre-populated, must be cleared
  ASSERT_EQ(parse_ampp_writable_slot_mask("", &slots), RSMI_STATUS_SUCCESS);
  EXPECT_TRUE(slots.empty());
}

TEST(GpuUnit, AmppWritableSlotMaskParseZeroMaskMeansNothingWritable) {
  std::vector<uint32_t> slots;
  ASSERT_EQ(parse_ampp_writable_slot_mask("0x0", &slots), RSMI_STATUS_SUCCESS);
  EXPECT_TRUE(slots.empty());
}

TEST(GpuUnit, AmppWritableSlotMaskParseMissingPrefixIsRejected) {
  std::vector<uint32_t> slots;
  EXPECT_EQ(parse_ampp_writable_slot_mask("e0", &slots), RSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, AmppWritableSlotMaskParseNonHexCharactersAreRejected) {
  std::vector<uint32_t> slots;
  EXPECT_EQ(parse_ampp_writable_slot_mask("0xzz", &slots), RSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, AmppWritableSlotMaskParseTrailingGarbageIsRejected) {
  std::vector<uint32_t> slots;
  EXPECT_EQ(parse_ampp_writable_slot_mask("0xe0garbage", &slots), RSMI_STATUS_UNEXPECTED_DATA);
}

// The slot count is a runtime driver property, so a mask describing more
// than today's 8 slots must parse, not hard-fail on a compile-time ceiling.
TEST(GpuUnit, AmppWritableSlotMaskParseBitsAboveEightSlotsAreAccepted) {
  std::vector<uint32_t> slots;
  ASSERT_EQ(parse_ampp_writable_slot_mask("0xe000", &slots), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(slots, (std::vector<uint32_t>{13, 14, 15}));
}

TEST(GpuUnit, AmppWritableSlotMaskParseHighestBitAccepted) {
  std::vector<uint32_t> slots;
  ASSERT_EQ(parse_ampp_writable_slot_mask("0x8000000000000000", &slots), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(slots, (std::vector<uint32_t>{63}));
}

TEST(GpuUnit, AmppWritableSlotMaskParseOverflowedValueIsRejected) {
  std::vector<uint32_t> slots;
  EXPECT_EQ(parse_ampp_writable_slot_mask("0xffffffffffffffffff", &slots),
            RSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, AmppWritableSlotMaskParsePrefixOnlyIsRejected) {
  std::vector<uint32_t> slots;
  EXPECT_EQ(parse_ampp_writable_slot_mask("0x", &slots), RSMI_STATUS_UNEXPECTED_DATA);
}

}  // namespace
