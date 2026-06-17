// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/perfetto/packet_framing.hpp"

#include <cstdint>
#include <vector>

using rocprofsys::core::append_varint;
using rocprofsys::core::read_varint;
using rocprofsys::core::rewrite_trace_packet;
using rocprofsys::core::rewrite_trace_packet_checked;
using rocprofsys::core::rewrite_trace_packet_status;
using rocprofsys::core::TRACE_PACKETS_TAG;
using rocprofsys::core::TRUSTED_SEQ_ID_TAG;

// ----------------------------------------------------------------------------
// read_varint / append_varint
// ----------------------------------------------------------------------------

TEST(packet_framing_varint, single_byte_round_trip)
{
    std::vector<char> buf;
    append_varint(buf, 0);
    append_varint(buf, 1);
    append_varint(buf, 127);
    EXPECT_EQ(buf.size(), 3u);

    std::size_t   pos = 0;
    std::uint64_t v   = 99;
    ASSERT_TRUE(read_varint(buf.data(), buf.size(), pos, v));
    EXPECT_EQ(v, 0u);
    ASSERT_TRUE(read_varint(buf.data(), buf.size(), pos, v));
    EXPECT_EQ(v, 1u);
    ASSERT_TRUE(read_varint(buf.data(), buf.size(), pos, v));
    EXPECT_EQ(v, 127u);
    EXPECT_EQ(pos, 3u);
}

TEST(packet_framing_varint, multi_byte_round_trip)
{
    std::vector<char> buf;
    append_varint(buf, 128);         // 2 bytes
    append_varint(buf, 16384);       // 3 bytes
    append_varint(buf, 1ULL << 35);  // 6 bytes

    std::size_t   pos = 0;
    std::uint64_t v   = 0;
    ASSERT_TRUE(read_varint(buf.data(), buf.size(), pos, v));
    EXPECT_EQ(v, 128u);
    ASSERT_TRUE(read_varint(buf.data(), buf.size(), pos, v));
    EXPECT_EQ(v, 16384u);
    ASSERT_TRUE(read_varint(buf.data(), buf.size(), pos, v));
    EXPECT_EQ(v, 1ULL << 35);
}

TEST(packet_framing_varint, max_uint64_uses_ten_bytes)
{
    std::vector<char> buf;
    append_varint(buf, UINT64_MAX);
    EXPECT_EQ(buf.size(), 10u);

    std::size_t   pos = 0;
    std::uint64_t v   = 0;
    ASSERT_TRUE(read_varint(buf.data(), buf.size(), pos, v));
    EXPECT_EQ(v, UINT64_MAX);
    EXPECT_EQ(pos, 10u);
}

TEST(packet_framing_varint, truncated_input_returns_false)
{
    const char    bytes[] = { static_cast<char>(
        0x80) };  // continuation bit set, no follow-up
    std::size_t   pos     = 0;
    std::uint64_t v       = 999;
    EXPECT_FALSE(read_varint(bytes, sizeof(bytes), pos, v));
}

TEST(packet_framing_varint, overlong_varint_returns_false)
{
    // 11+ continuation bytes would overflow 64-bit accumulator.
    std::vector<char> bytes(11, static_cast<char>(0x80));
    bytes.push_back(static_cast<char>(0x01));
    std::size_t   pos = 0;
    std::uint64_t v   = 0;
    EXPECT_FALSE(read_varint(bytes.data(), bytes.size(), pos, v));
}

TEST(packet_framing_varint, ten_byte_payload_overflow_returns_false)
{
    std::vector<char> bytes(9, static_cast<char>(0x80));
    bytes.push_back(static_cast<char>(0x02));

    std::size_t   pos = 0;
    std::uint64_t v   = 0;
    EXPECT_FALSE(read_varint(bytes.data(), bytes.size(), pos, v));
}

// ----------------------------------------------------------------------------
// rewrite_trace_packet
// ----------------------------------------------------------------------------

namespace
{
// Builds a TracePacket payload with the SDK-stamped placeholder
// trusted_packet_sequence_id (field 10) = 1, plus a synthetic
// timestamp field 8 (varint) carrying `ts` and an opaque payload
// field 11 (length-delimited bytes) carrying `payload`. Used to
// verify that rewrite_trace_packet swaps the seq_id without
// disturbing the other fields. Field numbers stay <= 15 so each
// tag is a single varint byte.
std::vector<char>
build_packet_with_placeholder_seq_id(std::uint64_t ts, const std::vector<char>& payload)
{
    std::vector<char> p;
    // field 8 (timestamp_ns), wire 0 (varint), value = ts
    p.push_back(static_cast<char>((8 << 3) | 0));
    append_varint(p, ts);
    // field 10 (trusted_packet_sequence_id), wire 0, placeholder 1
    p.push_back(static_cast<char>(TRUSTED_SEQ_ID_TAG));
    append_varint(p, 1);
    // field 11, wire 2 (length-delimited), payload
    p.push_back(static_cast<char>((11 << 3) | 2));
    append_varint(p, payload.size());
    p.insert(p.end(), payload.begin(), payload.end());
    return p;
}

// Decodes a Trace.packets-framed buffer into the raw payload of the
// first inner TracePacket. Asserts the framing tag and length match.
std::vector<char>
extract_first_packet_payload(const std::vector<char>& framed)
{
    EXPECT_GE(framed.size(), 2u);
    EXPECT_EQ(static_cast<std::uint8_t>(framed[0]), TRACE_PACKETS_TAG);
    std::size_t   pos = 1;
    std::uint64_t len = 0;
    EXPECT_TRUE(read_varint(framed.data(), framed.size(), pos, len));
    EXPECT_LE(pos + len, framed.size());
    return std::vector<char>(framed.begin() + pos, framed.begin() + pos + len);
}
}  // namespace

TEST(packet_framing_rewrite, applies_offset_to_existing_seq_id)
{
    // Input seq_id is the SDK placeholder (1); offset 42 must produce
    // effective seq_id 43, preserving the original-seq_id contribution
    // rather than replacing it. The other fields stay byte-identical.
    auto in = build_packet_with_placeholder_seq_id(0xCAFE, { 'x', 'y', 'z' });

    std::vector<char> dst;
    ASSERT_TRUE(rewrite_trace_packet(dst, in.data(), in.size(), 42));

    auto inner = extract_first_packet_payload(dst);

    std::vector<char> expected;
    expected.push_back(static_cast<char>((8 << 3) | 0));
    append_varint(expected, 0xCAFE);
    expected.push_back(static_cast<char>((11 << 3) | 2));
    append_varint(expected, 3);
    expected.insert(expected.end(), { 'x', 'y', 'z' });
    expected.push_back(static_cast<char>(TRUSTED_SEQ_ID_TAG));
    append_varint(expected, 1u + 42u);

    EXPECT_EQ(inner, expected);
}

TEST(packet_framing_rewrite, preserves_distinct_seq_ids_across_packets)
{
    // Two packets from the same source carrying different original seq_ids
    // (5 and 9). With the same offset (100) applied, the effective seq_ids
    // stay distinct (105 and 109) so each retains its own interned-data
    // namespace in the merged stream.
    auto build_with = [](std::uint32_t seq_id) {
        std::vector<char> p;
        p.push_back(static_cast<char>(TRUSTED_SEQ_ID_TAG));
        append_varint(p, seq_id);
        return p;
    };

    auto in_5 = build_with(5);
    auto in_9 = build_with(9);

    std::vector<char> dst;
    ASSERT_TRUE(rewrite_trace_packet(dst, in_5.data(), in_5.size(), 100));
    ASSERT_TRUE(rewrite_trace_packet(dst, in_9.data(), in_9.size(), 100));

    auto extract_seq_id_at = [&](std::size_t start) -> std::uint64_t {
        EXPECT_EQ(static_cast<std::uint8_t>(dst[start]), TRACE_PACKETS_TAG);
        std::size_t   pos = start + 1;
        std::uint64_t len = 0;
        EXPECT_TRUE(read_varint(dst.data(), dst.size(), pos, len));
        const auto payload_start = pos;
        EXPECT_EQ(static_cast<std::uint8_t>(dst[payload_start]), TRUSTED_SEQ_ID_TAG);
        pos                  = payload_start + 1;
        std::uint64_t seq_id = 0;
        EXPECT_TRUE(read_varint(dst.data(), dst.size(), pos, seq_id));
        EXPECT_EQ(pos, payload_start + static_cast<std::size_t>(len));
        return seq_id;
    };

    EXPECT_EQ(extract_seq_id_at(0), 105u);
    // Second packet starts after first frame (tag + len varint + payload).
    std::size_t   pos = 1;
    std::uint64_t len = 0;
    ASSERT_TRUE(read_varint(dst.data(), dst.size(), pos, len));
    const auto second_start = pos + static_cast<std::size_t>(len);
    EXPECT_EQ(extract_seq_id_at(second_start), 109u);
}

TEST(packet_framing_rewrite, distinct_offsets_for_separate_calls)
{
    auto in = build_packet_with_placeholder_seq_id(1, {});

    std::vector<char> dst_7;
    std::vector<char> dst_99;
    ASSERT_TRUE(rewrite_trace_packet(dst_7, in.data(), in.size(), 7));
    ASSERT_TRUE(rewrite_trace_packet(dst_99, in.data(), in.size(), 99));

    EXPECT_NE(dst_7, dst_99);
}

TEST(packet_framing_rewrite, offset_zero_leaves_seq_id_unchanged)
{
    // Regression guard: offset 0 must not be a sentinel for "no offset
    // applied" — it must add zero and emit the original seq_id.
    auto in = build_packet_with_placeholder_seq_id(0xBEEF, { 'q' });

    std::vector<char> dst;
    ASSERT_TRUE(rewrite_trace_packet(dst, in.data(), in.size(), 0));

    auto inner = extract_first_packet_payload(dst);

    std::vector<char> expected;
    expected.push_back(static_cast<char>((8 << 3) | 0));
    append_varint(expected, 0xBEEF);
    expected.push_back(static_cast<char>((11 << 3) | 2));
    append_varint(expected, 1);
    expected.push_back('q');
    expected.push_back(static_cast<char>(TRUSTED_SEQ_ID_TAG));
    append_varint(expected, 1u);

    EXPECT_EQ(inner, expected);
}

TEST(packet_framing_rewrite, checked_rewrite_rejects_seq_id_outside_limit)
{
    auto in = build_packet_with_placeholder_seq_id(0xBEEF, { 'q' });

    std::vector<char> dst;
    EXPECT_EQ(rewrite_trace_packet_checked(dst, in.data(), in.size(), 7, 8),
              rewrite_trace_packet_status::seq_id_out_of_range);
    EXPECT_TRUE(dst.empty()) << "failed checked rewrite must not append partial output";
}

TEST(packet_framing_rewrite, checked_rewrite_accepts_seq_id_below_limit)
{
    auto in = build_packet_with_placeholder_seq_id(0xBEEF, { 'q' });

    std::vector<char> dst;
    EXPECT_EQ(rewrite_trace_packet_checked(dst, in.data(), in.size(), 7, 9),
              rewrite_trace_packet_status::success);
    EXPECT_FALSE(dst.empty());
}

TEST(packet_framing_rewrite, rejects_truncated_varint_tag)
{
    const char        bytes[] = { static_cast<char>(0x80) };  // truncated tag
    std::vector<char> dst;
    EXPECT_FALSE(rewrite_trace_packet(dst, bytes, sizeof(bytes), 1));
}

TEST(packet_framing_rewrite, rejects_truncated_length_delimited_field)
{
    // tag for field 50 wire 2, then a length that exceeds remaining bytes.
    std::vector<char> bytes;
    bytes.push_back(static_cast<char>((50 << 3) | 2));
    append_varint(bytes, 100);  // claims 100 bytes follow, but we provide none
    std::vector<char> dst;
    EXPECT_FALSE(rewrite_trace_packet(dst, bytes.data(), bytes.size(), 1));
}

TEST(packet_framing_rewrite, rejects_unknown_wire_type)
{
    // Wire type 3 (start group, deprecated) is not supported.
    std::vector<char> bytes;
    bytes.push_back(static_cast<char>((1 << 3) | 3));
    std::vector<char> dst;
    EXPECT_FALSE(rewrite_trace_packet(dst, bytes.data(), bytes.size(), 1));
}

TEST(packet_framing_rewrite, fixed64_wire_advances_eight_bytes)
{
    // field 1 wire 1 (64-bit fixed), 8 bytes of payload
    std::vector<char> bytes;
    bytes.push_back(static_cast<char>((1 << 3) | 1));
    for(int i = 0; i < 8; ++i)
        bytes.push_back(static_cast<char>(0xAA));

    std::vector<char> dst;
    ASSERT_TRUE(rewrite_trace_packet(dst, bytes.data(), bytes.size(), 5));

    auto inner = extract_first_packet_payload(dst);
    EXPECT_GE(inner.size(), 9u);  // 1 tag + 8 fixed + seq_id pair
}

TEST(packet_framing_rewrite, rejects_truncated_fixed64_field)
{
    std::vector<char> bytes;
    bytes.push_back(static_cast<char>((1 << 3) | 1));
    for(int i = 0; i < 7; ++i)
        bytes.push_back(static_cast<char>(0xAA));

    std::vector<char> dst;
    EXPECT_FALSE(rewrite_trace_packet(dst, bytes.data(), bytes.size(), 5));
}

TEST(packet_framing_rewrite, fixed32_wire_advances_four_bytes)
{
    // field 1 wire 5 (32-bit fixed), 4 bytes of payload
    std::vector<char> bytes;
    bytes.push_back(static_cast<char>((1 << 3) | 5));
    for(int i = 0; i < 4; ++i)
        bytes.push_back(static_cast<char>(0xBB));

    std::vector<char> dst;
    EXPECT_TRUE(rewrite_trace_packet(dst, bytes.data(), bytes.size(), 5));
}

TEST(packet_framing_rewrite, rejects_truncated_fixed32_field)
{
    std::vector<char> bytes;
    bytes.push_back(static_cast<char>((1 << 3) | 5));
    for(int i = 0; i < 3; ++i)
        bytes.push_back(static_cast<char>(0xBB));

    std::vector<char> dst;
    EXPECT_FALSE(rewrite_trace_packet(dst, bytes.data(), bytes.size(), 5));
}

TEST(packet_framing_rewrite, empty_input_emits_offset_as_seq_id)
{
    // No input bytes means no original seq_id is observed; the rewritten
    // payload contains just the offset value as the new seq_id.
    std::vector<char> dst;
    ASSERT_TRUE(rewrite_trace_packet(dst, nullptr, 0, 13));

    auto              inner = extract_first_packet_payload(dst);
    std::vector<char> expected;
    expected.push_back(static_cast<char>(TRUSTED_SEQ_ID_TAG));
    append_varint(expected, 13);
    EXPECT_EQ(inner, expected);
}
