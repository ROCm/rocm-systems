// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt_provenance_test.cpp
/// @brief Wire-format tests for the DBT translation provenance note.

#include "rocjitsu/code/dbt/dbt_provenance.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rocjitsu {
namespace {

TEST(DbtProvenance, RoundTripsB0ToA0) {
  const DbtProvenance input{.input_revision = ProcessorRevision::Gfx1250B0,
                            .output_revision = ProcessorRevision::Gfx1250A0};
  const auto bytes = serialize_dbt_provenance(input);
  ASSERT_FALSE(bytes.empty());

  const auto parsed = parse_dbt_provenance(bytes);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->input_revision, ProcessorRevision::Gfx1250B0);
  EXPECT_EQ(parsed->output_revision, ProcessorRevision::Gfx1250A0);
}

TEST(DbtProvenance, DirectionIsPreservedNotNormalized) {
  // The note records exactly what was translated; a reverse direction must decode
  // as the reverse, so a reader can distinguish it from the expected direction.
  const DbtProvenance reverse{.input_revision = ProcessorRevision::Gfx1250A0,
                              .output_revision = ProcessorRevision::Gfx1250B0};
  const auto parsed = parse_dbt_provenance(serialize_dbt_provenance(reverse));
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->input_revision, ProcessorRevision::Gfx1250A0);
  EXPECT_EQ(parsed->output_revision, ProcessorRevision::Gfx1250B0);
}

TEST(DbtProvenance, RefusesToEmitUnspecifiedRevision) {
  EXPECT_TRUE(serialize_dbt_provenance({.input_revision = ProcessorRevision::Unspecified,
                                        .output_revision = ProcessorRevision::Gfx1250A0})
                  .empty());
  EXPECT_TRUE(serialize_dbt_provenance({.input_revision = ProcessorRevision::Gfx1250B0,
                                        .output_revision = ProcessorRevision::Unspecified})
                  .empty());
}

TEST(DbtProvenance, HeaderBytesPinMagicAndVersion) {
  const auto bytes = serialize_dbt_provenance({.input_revision = ProcessorRevision::Gfx1250B0,
                                               .output_revision = ProcessorRevision::Gfx1250A0});
  ASSERT_EQ(bytes.size(), 24u);
  constexpr std::array<uint8_t, 8> expected_magic = {'R', 'J', 'P', 'R', 'O', 'V', '1', '\0'};
  EXPECT_TRUE(std::equal(expected_magic.begin(), expected_magic.end(), bytes.begin()));
  uint32_t version = 0;
  std::memcpy(&version, bytes.data() + 8, sizeof(version));
  EXPECT_EQ(version, 1u);
}

TEST(DbtProvenance, RejectsUnknownWireVersion) {
  auto bytes = serialize_dbt_provenance({.input_revision = ProcessorRevision::Gfx1250B0,
                                         .output_revision = ProcessorRevision::Gfx1250A0});
  const uint32_t unknown_version = 2;
  std::memcpy(bytes.data() + 8, &unknown_version, sizeof(unknown_version));
  EXPECT_FALSE(parse_dbt_provenance(bytes).has_value());
}

TEST(DbtProvenance, RejectsTruncatedHeader) {
  auto bytes = serialize_dbt_provenance({.input_revision = ProcessorRevision::Gfx1250B0,
                                         .output_revision = ProcessorRevision::Gfx1250A0});
  ASSERT_EQ(bytes.size(), 24u);
  bytes.resize(23);
  EXPECT_FALSE(parse_dbt_provenance(bytes).has_value());
}

TEST(DbtProvenance, RejectsNonZeroReservedField) {
  auto bytes = serialize_dbt_provenance({.input_revision = ProcessorRevision::Gfx1250B0,
                                         .output_revision = ProcessorRevision::Gfx1250A0});
  // reserved lives at header offset 20; a non-zero value keeps the field free for
  // a forward-compatible extension and must be rejected today.
  const uint32_t bogus_reserved = 1;
  std::memcpy(bytes.data() + 20, &bogus_reserved, sizeof(bogus_reserved));
  EXPECT_FALSE(parse_dbt_provenance(bytes).has_value());
}

TEST(DbtProvenance, RejectsUnknownRevisionEncoding) {
  auto bytes = serialize_dbt_provenance({.input_revision = ProcessorRevision::Gfx1250B0,
                                         .output_revision = ProcessorRevision::Gfx1250A0});
  // input_revision wire value lives at header offset 12. An unrecognized code is
  // treated as malformed rather than silently mapped to a revision.
  const uint32_t bogus_revision = 99;
  std::memcpy(bytes.data() + 12, &bogus_revision, sizeof(bogus_revision));
  EXPECT_FALSE(parse_dbt_provenance(bytes).has_value());
}

TEST(DbtProvenance, RejectsUnspecifiedRevisionEncoding) {
  auto bytes = serialize_dbt_provenance({.input_revision = ProcessorRevision::Gfx1250B0,
                                         .output_revision = ProcessorRevision::Gfx1250A0});
  // A wire value of 0 (Unspecified) is never legitimately emitted, so decoding one
  // must be rejected -- this keeps a zeroed buffer from parsing as a valid note.
  const uint32_t unspecified = 0;
  std::memcpy(bytes.data() + 12, &unspecified, sizeof(unspecified));
  EXPECT_FALSE(parse_dbt_provenance(bytes).has_value());
}

} // namespace
} // namespace rocjitsu
