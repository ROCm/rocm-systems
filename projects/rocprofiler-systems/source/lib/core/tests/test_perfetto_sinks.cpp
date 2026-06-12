// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output_file_registry.hpp"
#include "core/perfetto/locked_file_append.hpp"
#include "core/perfetto/packet_framing.hpp"
#include "core/perfetto/sinks.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

TEST(recording_sink, default_state_is_empty_and_unfinalized)
{
    rocprofsys::core::recording_sink sink;
    EXPECT_TRUE(sink.records().empty());
    EXPECT_FALSE(sink.finalized());
}

TEST(recording_sink, on_source_drained_captures_records_in_order)
{
    rocprofsys::core::recording_sink sink;

    sink.on_source_drained(11, std::vector<char>{ 'a', 'b', 'c' });
    sink.on_source_drained(22, std::vector<char>{ 'x', 'y' });

    ASSERT_EQ(sink.records().size(), 2u);
    EXPECT_EQ(sink.records()[0].first, 11);
    EXPECT_EQ(sink.records()[0].second, (std::vector<char>{ 'a', 'b', 'c' }));
    EXPECT_EQ(sink.records()[1].first, 22);
    EXPECT_EQ(sink.records()[1].second, (std::vector<char>{ 'x', 'y' }));

    EXPECT_FALSE(sink.finalized()) << "finalize must not auto-fire on drain";
}

TEST(recording_sink, finalize_sets_flag)
{
    rocprofsys::core::recording_sink sink;
    sink.on_source_drained(0, std::vector<char>{ 'z' });
    EXPECT_FALSE(sink.finalized());

    sink.finalize();
    EXPECT_TRUE(sink.finalized());
}

TEST(recording_sink, finalize_without_drain_is_safe)
{
    // Drain contract allows finalize-after-zero-sources (e.g. empty pid set).
    rocprofsys::core::recording_sink sink;
    EXPECT_NO_THROW(sink.finalize());
    EXPECT_TRUE(sink.finalized());
    EXPECT_TRUE(sink.records().empty());
}

// ----------------------------------------------------------------------------
// locked_file_append
// ----------------------------------------------------------------------------

TEST(locked_file_append, creates_parent_directory_and_appends_in_order)
{
    const auto root = std::filesystem::path{ ::testing::TempDir() } /
                      "rocprofsys-locked-file-append-test";
    const auto path = root / "nested" / "merged.proto";

    std::filesystem::remove_all(root);

    EXPECT_EQ(rocprofsys::core::append_with_file_lock(path.string(), "abc", 3),
              rocprofsys::core::locked_append_status::success);
    EXPECT_EQ(rocprofsys::core::append_with_file_lock(path.string(), "def", 3),
              rocprofsys::core::locked_append_status::success);

    std::ifstream     ifs{ path, std::ios::binary };
    const std::string contents{ std::istreambuf_iterator<char>{ ifs },
                                std::istreambuf_iterator<char>{} };
    EXPECT_EQ(contents, "abcdef");

    std::filesystem::remove_all(root);
}

TEST(locked_file_append, empty_filename_reports_open_failed)
{
    EXPECT_EQ(rocprofsys::core::append_with_file_lock("", "x", 1),
              rocprofsys::core::locked_append_status::open_failed);
}

TEST(locked_file_append, status_name_handles_known_and_unknown_values)
{
    using rocprofsys::core::locked_append_status;

    EXPECT_EQ(rocprofsys::core::status_name(locked_append_status::success), "success");
    EXPECT_EQ(rocprofsys::core::status_name(locked_append_status::open_failed),
              "open_failed");
    EXPECT_EQ(rocprofsys::core::status_name(locked_append_status::lock_failed),
              "lock_failed");
    EXPECT_EQ(rocprofsys::core::status_name(locked_append_status::write_failed),
              "write_failed");
    EXPECT_EQ(rocprofsys::core::status_name(static_cast<locked_append_status>(999)),
              "unknown");
}

// ----------------------------------------------------------------------------
// per_pid_file_sink
// ----------------------------------------------------------------------------

TEST(per_pid_file_sink, empty_bytes_is_early_return)
{
    // Empty drains must not touch the filesystem or the registry —
    // per_pid_file_sink::on_source_drained returns early on empty bytes
    // so the (uninitialised in unit tests) config singleton is never
    // queried for the output filename.
    rocprofsys::output_file_registry    registry;
    rocprofsys::core::per_pid_file_sink sink{ static_cast<pid_t>(1), registry };

    EXPECT_NO_THROW(sink.on_source_drained(1, std::vector<char>{}));
    EXPECT_NO_THROW(sink.finalize());
}

// File-IO and per-pid open-failure isolation coverage for per_pid_file_sink
// is exercised by the integration tests in tests/pytest/. Unit-level
// mocking of config::get_perfetto_output_filename(...) here would require
// library-init machinery this test binary does not own.

// ----------------------------------------------------------------------------
// single_file_sink
// ----------------------------------------------------------------------------

namespace
{
using rocprofsys::core::append_varint;
using rocprofsys::core::read_varint;
using rocprofsys::core::TRACE_PACKETS_TAG;
using rocprofsys::core::TRUSTED_SEQ_ID_TAG;

// Builds a Trace.packets-framed buffer containing one TracePacket whose
// trusted_packet_sequence_id is the SDK placeholder (1) and whose payload
// is a single length-delimited field 11 carrying `marker`. Matches the
// shape that single_file_sink::on_source_drained walks.
std::vector<char>
build_framed_placeholder_packet(char marker)
{
    std::vector<char> inner;
    inner.push_back(static_cast<char>(TRUSTED_SEQ_ID_TAG));
    append_varint(inner, 1);
    inner.push_back(static_cast<char>((11 << 3) | 2));
    append_varint(inner, 1);
    inner.push_back(marker);

    std::vector<char> framed;
    framed.push_back(static_cast<char>(TRACE_PACKETS_TAG));
    append_varint(framed, inner.size());
    framed.insert(framed.end(), inner.begin(), inner.end());
    return framed;
}

// Decodes the seq_id and payload-marker of the framed packet at `start`
// in `buf`, returning the byte index immediately after that packet.
std::size_t
read_framed_packet(const std::vector<char>& buf, std::size_t start,
                   std::uint64_t& out_seq_id, char& out_marker)
{
    EXPECT_EQ(static_cast<std::uint8_t>(buf[start]), TRACE_PACKETS_TAG);
    std::size_t   pos = start + 1;
    std::uint64_t len = 0;
    EXPECT_TRUE(read_varint(buf.data(), buf.size(), pos, len));
    const auto payload_start = pos;
    const auto payload_end   = payload_start + static_cast<std::size_t>(len);

    out_seq_id = 0;
    out_marker = 0;
    while(pos < payload_end)
    {
        std::uint64_t tag = 0;
        EXPECT_TRUE(read_varint(buf.data(), buf.size(), pos, tag));
        if(tag == TRUSTED_SEQ_ID_TAG)
        {
            EXPECT_TRUE(read_varint(buf.data(), buf.size(), pos, out_seq_id));
        }
        else if(tag == ((11 << 3) | 2))
        {
            std::uint64_t field_len = 0;
            EXPECT_TRUE(read_varint(buf.data(), buf.size(), pos, field_len));
            EXPECT_EQ(field_len, 1u);
            out_marker = buf[pos];
            pos += static_cast<std::size_t>(field_len);
        }
        else
        {
            ADD_FAILURE() << "unexpected field tag in test packet: " << tag;
            return payload_end;
        }
    }
    return payload_end;
}
}  // namespace

TEST(single_file_sink, cross_source_preserves_seq_id_namespace)
{
    // Feed two sources whose inputs both carry the SDK placeholder
    // seq_id=1. Each source must end up with its own disjoint effective
    // seq_id so downstream interned-data resolution does not collapse
    // the two sources' iid namespaces into one.
    rocprofsys::output_file_registry   registry;
    rocprofsys::core::single_file_sink sink{ registry };

    auto bytes_a = build_framed_placeholder_packet('A');
    auto bytes_b = build_framed_placeholder_packet('B');

    sink.on_source_drained(101, std::move(bytes_a));
    sink.on_source_drained(202, std::move(bytes_b));

    const auto& buf = sink.buffer_for_testing();
    ASSERT_FALSE(buf.empty());

    std::uint64_t seq_id_a = 0;
    char          marker_a = 0;
    const auto    after_a  = read_framed_packet(buf, 0, seq_id_a, marker_a);

    std::uint64_t seq_id_b = 0;
    char          marker_b = 0;
    read_framed_packet(buf, after_a, seq_id_b, marker_b);

    EXPECT_EQ(marker_a, 'A');
    EXPECT_EQ(marker_b, 'B');

    // Distinct sources must land in disjoint seq_id ranges. The exact
    // base values are an implementation detail (set by the per-source
    // stride), but the difference must be at least one stride so the
    // ranges cannot overlap on any subsequent packets.
    ASSERT_GT(seq_id_b, seq_id_a);
    EXPECT_GE(seq_id_b - seq_id_a, 1u << 16);
}

TEST(single_file_sink, same_source_shares_base_offset)
{
    // Two drains from the same source share the same base offset, so
    // their outputs end up with the same effective seq_id (when their
    // original seq_ids match). The per-source allocation is sticky.
    rocprofsys::output_file_registry   registry;
    rocprofsys::core::single_file_sink sink{ registry };

    sink.on_source_drained(7, build_framed_placeholder_packet('X'));
    sink.on_source_drained(7, build_framed_placeholder_packet('Y'));

    const auto& buf = sink.buffer_for_testing();
    ASSERT_FALSE(buf.empty());

    std::uint64_t seq_id_x = 0;
    char          marker_x = 0;
    const auto    after_x  = read_framed_packet(buf, 0, seq_id_x, marker_x);

    std::uint64_t seq_id_y = 0;
    char          marker_y = 0;
    read_framed_packet(buf, after_x, seq_id_y, marker_y);

    EXPECT_EQ(marker_x, 'X');
    EXPECT_EQ(marker_y, 'Y');
    EXPECT_EQ(seq_id_x, seq_id_y);
}

TEST(single_file_sink, malformed_trace_packets_tag_drops_remainder)
{
    rocprofsys::output_file_registry   registry;
    rocprofsys::core::single_file_sink sink{ registry };

    auto bytes = build_framed_placeholder_packet('A');
    bytes.push_back(static_cast<char>(0xFF));
    auto ignored = build_framed_placeholder_packet('B');
    bytes.insert(bytes.end(), ignored.begin(), ignored.end());

    sink.on_source_drained(9, std::move(bytes));

    const auto& buf = sink.buffer_for_testing();
    ASSERT_FALSE(buf.empty());

    std::uint64_t seq_id = 0;
    char          marker = 0;
    const auto    end    = read_framed_packet(buf, 0, seq_id, marker);
    EXPECT_EQ(marker, 'A');
    EXPECT_EQ(end, buf.size()) << "packets after malformed tag must be dropped";
}

TEST(single_file_sink, truncated_trace_packets_frame_drops_remainder)
{
    rocprofsys::output_file_registry   registry;
    rocprofsys::core::single_file_sink sink{ registry };

    auto bytes = build_framed_placeholder_packet('A');
    bytes.push_back(static_cast<char>(TRACE_PACKETS_TAG));
    append_varint(bytes, 100);
    bytes.push_back('x');
    auto ignored = build_framed_placeholder_packet('B');
    bytes.insert(bytes.end(), ignored.begin(), ignored.end());

    sink.on_source_drained(9, std::move(bytes));

    const auto& buf = sink.buffer_for_testing();
    ASSERT_FALSE(buf.empty());

    std::uint64_t seq_id = 0;
    char          marker = 0;
    const auto    end    = read_framed_packet(buf, 0, seq_id, marker);
    EXPECT_EQ(marker, 'A');
    EXPECT_EQ(end, buf.size()) << "packets after truncated frame must be dropped";
}

TEST(single_file_sink, malformed_inner_trace_packet_drops_remainder)
{
    rocprofsys::output_file_registry   registry;
    rocprofsys::core::single_file_sink sink{ registry };

    auto bytes = build_framed_placeholder_packet('A');
    bytes.push_back(static_cast<char>(TRACE_PACKETS_TAG));
    append_varint(bytes, 1);
    bytes.push_back(static_cast<char>((1 << 3) | 3));
    auto ignored = build_framed_placeholder_packet('B');
    bytes.insert(bytes.end(), ignored.begin(), ignored.end());

    sink.on_source_drained(9, std::move(bytes));

    const auto& buf = sink.buffer_for_testing();
    ASSERT_FALSE(buf.empty());

    std::uint64_t seq_id = 0;
    char          marker = 0;
    const auto    end    = read_framed_packet(buf, 0, seq_id, marker);
    EXPECT_EQ(marker, 'A');
    EXPECT_EQ(end, buf.size()) << "packets after malformed TracePacket must be dropped";
}
