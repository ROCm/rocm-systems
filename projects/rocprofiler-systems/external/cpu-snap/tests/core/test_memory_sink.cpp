// Unit tests for cpu_snap::core::memory_sink.

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <vector>

#include "cpu_snap/core/chunk_framer.hpp"
#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/reader.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/types.hpp"

namespace
{

using cpu_snap::core::chunk_framer;
using cpu_snap::core::memory_sink;

std::vector<std::byte>
make_chunk(std::size_t n_bytes, std::byte fill)
{
    return std::vector<std::byte>(n_bytes, fill);
}

}  // namespace

TEST(MemorySink, UnderLimitKeepsEveryChunk)
{
    memory_sink sink(memory_sink::config{ .max_bytes = 1024 });

    const auto chunk_a = make_chunk(100, std::byte{ 1 });
    const auto chunk_b = make_chunk(200, std::byte{ 2 });
    sink(chunk_a);
    sink(chunk_b);

    EXPECT_EQ(sink.bytes().size(), 300U);
    EXPECT_FALSE(sink.full());
    EXPECT_EQ(sink.dropped_chunks(), 0U);
    EXPECT_EQ(sink.dropped_bytes(), 0U);
}

TEST(MemorySink, OverLimitKeepsPrefixAndDropsRest)
{
    memory_sink sink(memory_sink::config{ .max_bytes = 150 });

    const auto chunk_a = make_chunk(100, std::byte{ 1 });  // fits: 100/150
    const auto chunk_b = make_chunk(80, std::byte{ 2 });   // would be 180 > 150: dropped
    const auto chunk_c = make_chunk(10, std::byte{ 3 });   // sink already full: dropped
    sink(chunk_a);
    sink(chunk_b);
    sink(chunk_c);

    EXPECT_EQ(sink.bytes().size(), 100U);
    EXPECT_TRUE(sink.full());
    EXPECT_EQ(sink.dropped_chunks(), 2U);
    EXPECT_EQ(sink.dropped_bytes(), 90U);
}

TEST(MemorySink, ExactFitDoesNotTriggerDrop)
{
    memory_sink sink(memory_sink::config{ .max_bytes = 100 });

    const auto chunk = make_chunk(100, std::byte{ 7 });
    sink(chunk);

    EXPECT_EQ(sink.bytes().size(), 100U);
    EXPECT_FALSE(sink.full());
    EXPECT_EQ(sink.dropped_chunks(), 0U);
}

TEST(MemorySink, KeptPrefixIsValidlyDecodableAfterDroppingLaterChunks)
{
    // Capture every physical chunk chunk_framer emits (file_header first,
    // then per-chunk record bodies) so the test knows the real chunk
    // boundaries instead of guessing byte offsets.
    std::vector<std::vector<std::byte>> captured_chunks;
    auto capture = [&captured_chunks](std::span<const std::byte> chunk) {
        captured_chunks.emplace_back(chunk.begin(), chunk.end());
    };

    chunk_framer::config cfg;
    cfg.chunk_bytes   = 64;  // small budget forces more than one physical chunk
    cfg.pid           = 42;
    cfg.start_unix_ns = 1234;
    cfg.process_name  = "memory_sink_test";

    chunk_framer framer(cfg, capture);
    framer.start();

    cpu_snap::core::new_node_record node;
    node.node_id   = 1;
    node.parent_id = 0;
    node.ip        = 0xABCD;
    framer.write(node);

    cpu_snap::core::sample_record sample_one;
    sample_one.timestamp_ns = 100;
    sample_one.tid          = 7;
    sample_one.leaf_node_id = 1;
    framer.write(sample_one);

    cpu_snap::core::sample_record sample_two;
    sample_two.timestamp_ns = 200;
    sample_two.tid          = 7;
    sample_two.leaf_node_id = 1;
    framer.write(sample_two);  // overflows the 64-byte budget into a new chunk
    framer.flush();

    ASSERT_GE(captured_chunks.size(), 3U)
        << "expected file_header + at least 2 record chunks";

    // Keep exactly [file_header, first record chunk]; every later chunk
    // must be dropped whole.
    const std::size_t kept_bytes = captured_chunks[0].size() + captured_chunks[1].size();
    memory_sink sink(memory_sink::config{ .max_bytes = kept_bytes });
    for(const auto& chunk : captured_chunks)
    {
        sink(chunk);
    }

    EXPECT_EQ(sink.bytes().size(), kept_bytes);
    EXPECT_TRUE(sink.full());
    EXPECT_EQ(sink.dropped_chunks(), captured_chunks.size() - 2);

    cpu_snap::core::reader reader;
    std::size_t            samples_seen = 0;
    reader.on_event([&](const cpu_snap::core::record_variant& rec) {
        if(cpu_snap::core::type_of(rec) == cpu_snap::core::record_type::sample)
        {
            ++samples_seen;
        }
    });
    reader.feed(sink.bytes());

    EXPECT_EQ(reader.counters().parse_errors, 0U);
    EXPECT_EQ(samples_seen, 1U)
        << "only the first SAMPLE's chunk was kept; the second was dropped whole";
}
