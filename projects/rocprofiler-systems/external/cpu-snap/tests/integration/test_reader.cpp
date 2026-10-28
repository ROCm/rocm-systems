// Unit tests for cpu_snap::core::reader.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cpu_snap/core/chunk_framer.hpp"
#include "cpu_snap/sampling/streaming_pipeline.hpp"
#include "cpu_snap/core/reader.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/types.hpp"
#include "support/stream_capture.hpp"

namespace
{

using namespace cpu_snap::core;
using cpu_snap::core::chunk_framer;
using cpu_snap::sampling::internal::streaming_pipeline;
using cpu_snap::test_support::byte_buffer;
using cpu_snap::test_support::make_framer_cfg;

struct event_recorder
{
    std::vector<record_variant> events;

    event_fn as_fn()
    {
        return [this](const record_variant& r) { events.push_back(r); };
    }

    template <typename T>
    [[nodiscard]] std::size_t count() const
    {
        std::size_t n = 0;
        for(const auto& r : events)
        {
            if(std::holds_alternative<T>(r)) ++n;
        }
        return n;
    }
};

streaming_pipeline::config
make_pipeline_cfg(sink_fn sink, std::size_t checkpoint_interval = 1000,
                  std::size_t   maps_poll_interval = 1000,
                  std::uint32_t chunk_bytes        = 1U << 16)
{
    streaming_pipeline::config cfg;
    cfg.sink                = std::move(sink);
    cfg.chunk_bytes         = chunk_bytes;
    cfg.flags               = 0x2;
    cfg.pid                 = 4242;
    cfg.start_unix_ns       = 1700000000000000000ULL;
    cfg.process_name        = "callstack_reader_test";
    cfg.maps_poll_interval  = maps_poll_interval;
    cfg.checkpoint_interval = checkpoint_interval;
    cfg.trie_max_nodes      = 0;
    return cfg;
}

/** Feed `bytes` to `r` one byte at a time. */
void
feed_byte_by_byte(reader& r, const std::vector<std::byte>& bytes)
{
    for(std::size_t i = 0; i < bytes.size(); ++i)
    {
        r.feed(std::span<const std::byte>(bytes.data() + i, 1));
    }
}

/** Feed `bytes` to `r` using a repeating slice pattern. */
void
feed_in_pattern(reader& r, const std::vector<std::byte>& bytes,
                std::initializer_list<std::size_t> pattern)
{
    std::size_t off = 0;
    const auto *        it  = pattern.begin();
    while(off < bytes.size())
    {
        std::size_t const n = std::min(*it, bytes.size() - off);
        r.feed(std::span<const std::byte>(bytes.data() + off, n));
        off += n;
        ++it;
        if(it == pattern.end()) it = pattern.begin();
    }
}

}  // namespace

// fatal magic mismatch

TEST(Reader, MagicMismatchSetsFatal)
{
    reader         r;
    event_recorder rec;
    r.on_event(rec.as_fn());

    std::vector<std::byte> bad(sizeof(file_header), std::byte{ 0xAB });
    r.feed(std::span<const std::byte>(bad.data(), bad.size()));

    EXPECT_TRUE(r.is_fatal());
    EXPECT_EQ(r.counters().parse_errors, 1U);
    EXPECT_EQ(rec.events.size(), 0U);

    // Subsequent feed is ignored once fatal.
    r.feed(std::span<const std::byte>(bad.data(), bad.size()));
    EXPECT_EQ(rec.events.size(), 0U);
}

// round-trip via chunk_framer directly

namespace
{

struct framer_roundtrip_fixture
{
    byte_buffer   buf;
    std::uint32_t leaf_id_of_chain = 0;

    void build()
    {
        chunk_framer fr(make_framer_cfg(4096), buf.as_sink());
        fr.start();

        // Build a 3-frame stack A -> B -> C (C is the root child) and a
        // sample referencing the leaf.
        new_node_record nn_c;
        nn_c.node_id   = 1;
        nn_c.parent_id = 0;
        nn_c.ip        = 0xC;
        new_node_record nn_b;
        nn_b.node_id   = 2;
        nn_b.parent_id = 1;
        nn_b.ip        = 0xB;
        new_node_record nn_a;
        nn_a.node_id     = 3;
        nn_a.parent_id   = 2;
        nn_a.ip          = 0xA;
        leaf_id_of_chain = nn_a.node_id;

        fr.write(nn_c);
        fr.write(nn_b);
        fr.write(nn_a);

        sample_record s;
        s.timestamp_ns = 100;
        s.tid          = 42;
        s.leaf_node_id = leaf_id_of_chain;
        fr.write(s);

        eof_record eof;
        eof.wall_ns               = 200;
        eof.total_samples_emitted = 1;
        fr.write(eof);

        fr.flush();
    }
};

}  // namespace

TEST(Reader, RoundTripSingleChunk)
{
    framer_roundtrip_fixture fx;
    fx.build();

    reader         r;
    event_recorder rec;
    r.on_event(rec.as_fn());
    r.feed(std::span<const std::byte>(fx.buf.bytes.data(), fx.buf.bytes.size()));

    EXPECT_FALSE(r.is_fatal());
    EXPECT_EQ(r.counters().parse_errors, 0U);
    EXPECT_EQ(rec.count<new_node_record>(), 3U);
    EXPECT_EQ(rec.count<sample_record>(), 1U);
    EXPECT_EQ(rec.count<eof_record>(), 1U);

    const auto chain = r.resolve_stack(fx.leaf_id_of_chain);
    ASSERT_EQ(chain.size(), 3U);
    EXPECT_EQ(chain[0], 0xAU);
    EXPECT_EQ(chain[1], 0xBU);
    EXPECT_EQ(chain[2], 0xCU);

    file_header h_copy{};
    std::memcpy(&h_copy, &r.header(), sizeof(h_copy));
    EXPECT_EQ(h_copy.pid, 1234U);
    EXPECT_EQ(h_copy.magic[0], 'C');
    EXPECT_EQ(h_copy.magic[3], 'P');
}

TEST(Reader, FeedByteByByte)
{
    framer_roundtrip_fixture fx;
    fx.build();

    reader         r;
    event_recorder rec;
    r.on_event(rec.as_fn());
    feed_byte_by_byte(r, fx.buf.bytes);

    EXPECT_EQ(r.counters().parse_errors, 0U);
    EXPECT_EQ(rec.count<new_node_record>(), 3U);
    EXPECT_EQ(rec.count<sample_record>(), 1U);
    EXPECT_EQ(rec.count<eof_record>(), 1U);

    const auto chain = r.resolve_stack(fx.leaf_id_of_chain);
    ASSERT_EQ(chain.size(), 3U);
    EXPECT_EQ(chain[0], 0xAU);
}

TEST(Reader, FeedInArbitraryChunks)
{
    framer_roundtrip_fixture fx;
    fx.build();

    reader         r;
    event_recorder rec;
    r.on_event(rec.as_fn());
    feed_in_pattern(r, fx.buf.bytes, { 17, 1, 53, 2, 7, 31 });

    EXPECT_EQ(r.counters().parse_errors, 0U);
    EXPECT_EQ(rec.count<new_node_record>(), 3U);
    EXPECT_EQ(rec.count<sample_record>(), 1U);
    EXPECT_EQ(rec.count<eof_record>(), 1U);
}

// chunk_seq gap

namespace
{

/**
 * Emit two chunks each holding one SAMPLE, then doctor the second
 * chunk's chunk_seq to skip a value (0 -> 2 instead of 0 -> 1).
 * Each chunk is header (16) + record (20) = 36 bytes.
 */
void
build_two_chunks_with_seq_gap(byte_buffer& buf)
{
    chunk_framer fr(make_framer_cfg(44), buf.as_sink());
    fr.start();

    sample_record s1;
    s1.timestamp_ns = 10;
    s1.tid          = 1;
    fr.write(s1);
    sample_record s2;
    s2.timestamp_ns = 20;
    s2.tid          = 2;
    fr.write(s2);
    fr.flush();

    // chunk_header layout: 'C''H', u16 version, u32 records_byte_count,
    // u64 chunk_seq. seq sits at offset 8 within the header.
    const std::size_t chunk1_start = sizeof(file_header) + 36U;
    const std::size_t seq_off      = chunk1_start + 8U;
    std::uint64_t     new_seq      = 2;
    std::memcpy(buf.bytes.data() + seq_off, &new_seq, sizeof(new_seq));
}

}  // namespace

TEST(Reader, ChunkSeqGapIncrementsCounter)
{
    byte_buffer buf;
    build_two_chunks_with_seq_gap(buf);

    reader         r;
    event_recorder rec;
    r.on_event(rec.as_fn());
    r.feed(std::span<const std::byte>(buf.bytes.data(), buf.bytes.size()));

    EXPECT_EQ(r.counters().gap_count, 1U);
    EXPECT_EQ(r.counters().last_seen_seq, 2U);
}

// unknown leaf

TEST(Reader, ResolveUnknownIdReturnsEmpty)
{
    reader const r;
    EXPECT_TRUE(r.resolve_stack(42).empty());
}

// streaming pipeline round-trip

TEST(Reader, PipelineRoundtrip)
{
    byte_buffer   buf;
    std::uint32_t expected_leaf    = 0;
    std::uint64_t expected_samples = 0;
    {
        streaming_pipeline pipe(
            make_pipeline_cfg(buf.as_sink(), /*ckpt*/ 100, /*maps*/ 10000));
        const std::uint64_t frames[] = { 0xAAA, 0xBBB, 0xCCC };
        for(std::uint64_t ts = 0; ts < 500; ++ts)
        {
            pipe.on_sample(7, ts, std::span<const std::uint64_t>(frames, 3));
        }
        pipe.on_lost(11);
        pipe.finalize();
        expected_samples = pipe.samples_emitted();
        // First sample's leaf id is the third (innermost) NEW_NODE.
        // streaming_pipeline assigns ids 1.. in NEW_NODE order, so leaf = 3.
        expected_leaf = 3;
    }

    reader         r;
    event_recorder rec;
    r.on_event(rec.as_fn());
    r.feed(std::span<const std::byte>(buf.bytes.data(), buf.bytes.size()));

    EXPECT_FALSE(r.is_fatal());
    EXPECT_EQ(r.counters().parse_errors, 0U);

    EXPECT_EQ(rec.count<sample_record>(), expected_samples);
    EXPECT_EQ(rec.count<new_node_record>(), 3U);
    EXPECT_EQ(rec.count<lost_record>(), 1U);
    EXPECT_EQ(rec.count<eof_record>(), 1U);
    EXPECT_GE(rec.count<checkpoint_record>(), 1U);
    EXPECT_EQ(rec.count<maps_init_record>(), 1U);

    const auto chain = r.resolve_stack(expected_leaf);
    ASSERT_EQ(chain.size(), 3U);
    EXPECT_EQ(chain[0], 0xAAAU);
    EXPECT_EQ(chain[1], 0xBBBU);
    EXPECT_EQ(chain[2], 0xCCCU);

    // Total event count matches the framer's record count.
    EXPECT_EQ(r.counters().records_emitted, rec.events.size());
}

TEST(Reader, ReaderHandlesStreamingFinalize)
{
    byte_buffer buf;
    {
        streaming_pipeline  pipe(make_pipeline_cfg(buf.as_sink(), 50, 10000));
        const std::uint64_t f[] = { 0x1, 0x2 };
        for(int i = 0; i < 60; ++i)
        {
            pipe.on_sample(1, i, std::span<const std::uint64_t>(f, 2));
        }
        pipe.finalize();
    }

    reader         r;
    event_recorder rec;
    r.on_event(rec.as_fn());
    r.feed(std::span<const std::byte>(buf.bytes.data(), buf.bytes.size()));

    const std::size_t samples = rec.count<sample_record>();
    const std::size_t nodes   = rec.count<new_node_record>();
    const std::size_t cks     = rec.count<checkpoint_record>();
    const std::size_t lost    = rec.count<lost_record>();
    const std::size_t maps_i  = rec.count<maps_init_record>();
    const std::size_t maps_d  = rec.count<maps_delta_record>();
    const std::size_t eof     = rec.count<eof_record>();

    EXPECT_EQ(samples, 60U);
    EXPECT_EQ(nodes, 2U);
    EXPECT_EQ(eof, 1U);
    EXPECT_EQ(maps_i, 1U);
    EXPECT_EQ(rec.events.size(), samples + nodes + cks + lost + maps_i + maps_d + eof);
}

