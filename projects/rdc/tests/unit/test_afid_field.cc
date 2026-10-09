/*
Copyright (c) 2025 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Pure-logic tests for the RDC_FI_AFID field added to decode AFID(s)+severity
// from the driver CPER ring. These require no hardware: they lock down the field
// registration contract that the rest of the daemon and the CLI (rdci dmon -e
// AFID) depend on. If AFID is not correctly registered in rdc_field.data /
// rdc_fields_supported, the field silently becomes unqueryable, so these guard
// the wiring rather than the CPER decode (which needs a GPU with RAS records and
// is covered by the hardware integration tests).

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "common/rdc_fields_supported.h"
#include "rdc/rdc.h"

namespace {

// The field id is part of the public ABI (used by rdci dmon -e 680 and stored in
// caches/records), so it must stay pinned at 680.
TEST(AfidField, EnumValueIsStable) { EXPECT_EQ(static_cast<uint32_t>(RDC_FI_AFID), 680u); }

// AFID must be a registered, valid field or the telemetry layer rejects it.
TEST(AfidField, IsValid) {
  EXPECT_TRUE(amd::rdc::is_field_valid(RDC_FI_AFID));
  EXPECT_FALSE(amd::rdc::is_field_valid(RDC_FI_INVALID));
}

// The descriptor table must carry AFID with its enum name, label, and a
// description, and be marked for display (rdci dmon -l relies on all of these).
TEST(AfidField, DescriptorMetadata) {
  auto& table = amd::rdc::get_field_id_description_from_id();
  auto it = table.find(static_cast<uint32_t>(RDC_FI_AFID));
  ASSERT_NE(it, table.end()) << "RDC_FI_AFID missing from descriptor table";

  EXPECT_EQ(it->second.enum_name, "RDC_FI_AFID");
  EXPECT_EQ(it->second.label, "AFID");
  EXPECT_FALSE(it->second.description.empty());
  EXPECT_TRUE(it->second.do_display);
}

// Name -> id resolution must round-trip both by enum name and by label, since the
// CLI accepts either form on the command line.
TEST(AfidField, NameLookupResolvesToId) {
  rdc_field_t id = RDC_FI_INVALID;

  ASSERT_TRUE(amd::rdc::get_field_id_from_name("RDC_FI_AFID", &id));
  EXPECT_EQ(id, RDC_FI_AFID);

  // The C wrapper used by the CLI must agree.
  EXPECT_EQ(get_field_id_from_name("RDC_FI_AFID"), RDC_FI_AFID);

  // An unknown name must not resolve to AFID (or anything valid).
  EXPECT_EQ(get_field_id_from_name("RDC_FI_NOT_A_FIELD"), RDC_FI_INVALID);
}

// field_id_string() (used throughout logging and the CLI) must return the label.
TEST(AfidField, FieldIdStringReturnsLabel) { EXPECT_STREQ(field_id_string(RDC_FI_AFID), "AFID"); }

// AFID sits between the ECC block (<= RDC_FI_ECC_LAST) and the XGMI block
// (>= 700); keep it in that reserved RAS/CPER gap so it collides with neither.
TEST(AfidField, IdInReservedRasGap) {
  EXPECT_GT(static_cast<uint32_t>(RDC_FI_AFID), static_cast<uint32_t>(RDC_FI_ECC_LAST));
  EXPECT_LT(static_cast<uint32_t>(RDC_FI_AFID), 700u);
}

}  // namespace
