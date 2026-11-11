// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/collection/sample_batch.hpp"
#include "core/trace_cache/archive.hpp"
#include "core/trace_cache/counter_samples.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace rocprofsys::trace_cache
{
namespace
{
constexpr std::uint32_t k_schema_id = 0x11223344U;
// above UINT32_MAX, so a 32-bit truncation is caught
constexpr std::uint64_t k_timestamp = 0x0102030405060708ULL;
constexpr double        k_value     = 42.5;
constexpr double        k_negative  = -0.125;

/** Bytes of a record as the buffer storage writes them. */
template <typename Record>
[[nodiscard]] std::vector<std::uint8_t>
to_bytes(const Record& record)
{
    std::vector<std::uint8_t> bytes(get_size(record));
    serialize(bytes.data(), record);
    return bytes;
}

/** Reads a record back and checks that every byte was consumed. */
template <typename Record>
[[nodiscard]] Record
from_bytes(std::vector<std::uint8_t> bytes)
{
    std::uint8_t* cursor = bytes.data();
    auto          record = deserialize<Record>(cursor);
    EXPECT_EQ(static_cast<std::size_t>(cursor - bytes.data()), bytes.size());
    return record;
}
}  // namespace

TEST(counter_records, type_ids_are_pinned)
{
    EXPECT_EQ(static_cast<std::uint32_t>(type_identifier_t::counter_batch), 0x000EU);
    EXPECT_EQ(static_cast<std::uint32_t>(type_identifier_t::gap_marker), 0x000FU);
    EXPECT_EQ(counter_batch_view::type_identifier, type_identifier_t::counter_batch);
    EXPECT_EQ(counter_batch::type_identifier, type_identifier_t::counter_batch);
    EXPECT_EQ(gap_marker::type_identifier, type_identifier_t::gap_marker);
}

TEST(counter_records, batch_view_reads_back_as_counter_batch_with_nan)
{
    const std::array<double, 4> row{ k_value, collection::k_no_value, k_negative,
                                     std::numeric_limits<double>::infinity() };
    const counter_batch_view    view{ .schema_id = k_schema_id,
                                      .timestamp = k_timestamp,
                                      .values    = row };

    const auto batch = from_bytes<counter_batch>(to_bytes(view));

    EXPECT_EQ(batch.schema_id, k_schema_id);
    EXPECT_EQ(batch.timestamp, k_timestamp);
    ASSERT_EQ(batch.values.size(), row.size());
    EXPECT_EQ(batch.values[0], k_value);
    EXPECT_TRUE(std::isnan(batch.values[1]));
    EXPECT_EQ(batch.values[2], k_negative);
    EXPECT_EQ(batch.values[3], row[3]);
}

TEST(counter_records, batch_view_and_counter_batch_share_the_wire_format)
{
    const std::array<double, 2> row{ k_value, k_negative };
    const counter_batch_view    view{ .schema_id = k_schema_id,
                                      .timestamp = k_timestamp,
                                      .values    = row };
    const counter_batch         batch{ .schema_id = k_schema_id,
                                       .timestamp = k_timestamp,
                                       .values    = { row.begin(), row.end() } };

    EXPECT_EQ(get_size(view), get_size(batch));
    EXPECT_EQ(to_bytes(view), to_bytes(batch));
}

TEST(counter_records, batch_without_values_round_trips)
{
    const counter_batch_view view{ .schema_id = k_schema_id,
                                   .timestamp = k_timestamp,
                                   .values    = std::span<const double>{} };

    const auto batch = from_bytes<counter_batch>(to_bytes(view));

    EXPECT_EQ(batch.schema_id, k_schema_id);
    EXPECT_TRUE(batch.values.empty());
}

TEST(counter_records, gap_marker_round_trips)
{
    constexpr std::uint64_t k_end = k_timestamp + 1;
    const gap_marker        gap{ .begin_ns = k_timestamp, .end_ns = k_end };

    const auto read = from_bytes<gap_marker>(to_bytes(gap));

    EXPECT_EQ(read.begin_ns, k_timestamp);
    EXPECT_EQ(read.end_ns, k_end);
    EXPECT_EQ(get_size(gap), 2 * sizeof(std::uint64_t));
}
}  // namespace rocprofsys::trace_cache
