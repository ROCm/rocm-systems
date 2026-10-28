// Unit tests for cpu_snap::sampling::internal::streaming_pipeline.
//
// Drives the pipeline from synthetic samples (no kernel involvement),
// captures sink output into a byte buffer, and decodes records inline.
// PR-8 will replace the hand-rolled decoder with the production reader.

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <sched.h>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/streaming_pipeline.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "cpu_snap/core/types.hpp"
#include "support/stream_capture.hpp"

namespace
{

using namespace cpu_snap::core;
using cpu_snap::sampling::internal::streaming_pipeline;
using cpu_snap::test_support::sink_capture;

/** Minimal cursor used by the local wire decoder. */
class wire_cursor
{
public:
    wire_cursor(const std::byte* data, std::size_t len) noexcept
    : m_data(data)
    , m_end(data + len)
    {}

    template <typename T>
    T read() noexcept
    {
        T v{};
        std::memcpy(&v, m_data, sizeof(T));
        m_data += sizeof(T);
        return v;
    }

    void read_into(void* dst, std::size_t n) noexcept
    {
        std::memcpy(dst, m_data, n);
        m_data += n;
    }

    void skip(std::size_t n) noexcept { m_data += n; }

    [[nodiscard]] std::size_t remaining() const noexcept
    {
        return static_cast<std::size_t>(m_end - m_data);
    }

    [[nodiscard]] const std::byte* ptr() const noexcept { return m_data; }

private:
    const std::byte* m_data;
    const std::byte* m_end;
};

/** Decoded record body: type tag + raw body bytes (post-header). */
struct decoded_record
{
    record_type            type = {};
    std::vector<std::byte> body;
};

/**
 * Parse the full captured stream (file_header + N chunks) into an
 * ordered list of records. Verifies framing along the way.
 */
[[nodiscard]] std::vector<decoded_record>
decode_stream(const sink_capture& sink)
{
    std::vector<decoded_record> out;
    if(sink.chunks.empty()) return out;

    // First span is the 64-byte file_header; record-bearing chunks
    // follow.
    EXPECT_EQ(sink.chunks[0].size(), sizeof(file_header));

    for(std::size_t ci = 1; ci < sink.chunks.size(); ++ci)
    {
        const auto& bytes = sink.chunks[ci];
        EXPECT_GE(bytes.size(), sizeof(chunk_header));

        wire_cursor  cur(bytes.data(), bytes.size());
        chunk_header ch{};
        cur.read_into(&ch, sizeof(ch));
        EXPECT_EQ(ch.magic[0], 'C');
        EXPECT_EQ(ch.magic[1], 'H');

        const std::size_t body_end = sizeof(chunk_header) + ch.records_byte_count;

        while(static_cast<std::size_t>(cur.ptr() - bytes.data()) < body_end)
        {
            const auto type_byte = cur.read<std::uint8_t>();
            cur.read<std::uint8_t>();  // reserved
            const auto body_size = cur.read<std::uint16_t>();

            decoded_record rec;
            rec.type = static_cast<record_type>(type_byte);
            rec.body.resize(body_size);
            if(body_size > 0)
            {
                cur.read_into(rec.body.data(), body_size);
            }
            out.push_back(std::move(rec));

            const std::uint32_t consumed = 4U + body_size;
            const std::uint32_t padded   = (consumed + 3U) & ~3U;
            cur.skip(padded - consumed);
        }
    }
    return out;
}

[[nodiscard]] new_node_record
decode_new_node(const decoded_record& r)
{
    EXPECT_EQ(r.type, record_type::new_node);
    EXPECT_EQ(r.body.size(), 16U);
    new_node_record nn{};
    std::memcpy(&nn, r.body.data(), sizeof(nn));
    return nn;
}

[[nodiscard]] sample_record
decode_sample(const decoded_record& r)
{
    EXPECT_EQ(r.type, record_type::sample);
    EXPECT_EQ(r.body.size(), 16U);
    sample_record s{};
    std::memcpy(&s, r.body.data(), sizeof(s));
    return s;
}

[[nodiscard]] lost_record
decode_lost(const decoded_record& r)
{
    EXPECT_EQ(r.type, record_type::lost);
    EXPECT_EQ(r.body.size(), 16U);
    lost_record lr{};
    std::memcpy(&lr, r.body.data(), sizeof(lr));
    return lr;
}

[[nodiscard]] eof_record
decode_eof(const decoded_record& r)
{
    EXPECT_EQ(r.type, record_type::eof);
    EXPECT_EQ(r.body.size(), 16U);
    eof_record e{};
    std::memcpy(&e, r.body.data(), sizeof(e));
    return e;
}

struct decoded_checkpoint
{
    std::uint64_t                         wall_ns;
    std::vector<per_tid_checkpoint_entry> entries;
};

[[nodiscard]] decoded_checkpoint
decode_checkpoint(const decoded_record& r)
{
    EXPECT_EQ(r.type, record_type::checkpoint);
    wire_cursor        cur(r.body.data(), r.body.size());
    decoded_checkpoint c{};
    c.wall_ns    = cur.read<std::uint64_t>();
    const auto n = cur.read<std::uint32_t>();
    cur.read<std::uint32_t>();  // padding word
    c.entries.resize(n);
    for(std::uint32_t i = 0; i < n; ++i)
    {
        per_tid_checkpoint_entry e{};
        e.tid        = cur.read<std::uint32_t>();
        e.leaf_id    = cur.read<std::uint32_t>();
        e.samples    = cur.read<std::uint64_t>();
        e.lost       = cur.read<std::uint64_t>();
        c.entries[i] = e;
    }
    return c;
}

streaming_pipeline::config
make_cfg(sink_fn sink, std::size_t checkpoint_interval = 1000,
         std::size_t maps_poll_interval = 1000, std::size_t trie_max_nodes = 0,
         std::uint32_t chunk_bytes = 1U << 16)
{
    streaming_pipeline::config cfg;
    cfg.sink                = std::move(sink);
    cfg.chunk_bytes         = chunk_bytes;
    cfg.flags               = 0x2;  // FP
    cfg.pid                 = 4242;
    cfg.start_unix_ns       = 1700000000000000000ULL;
    cfg.process_name        = "callstack_test";
    cfg.maps_poll_interval  = maps_poll_interval;
    cfg.checkpoint_interval = checkpoint_interval;
    cfg.trie_max_nodes      = trie_max_nodes;
    return cfg;
}

}  // namespace

// construction

TEST(StreamingPipeline, NullSinkThrows)
{
    streaming_pipeline::config cfg;
    cfg.sink         = nullptr;
    cfg.chunk_bytes  = 4096;
    cfg.process_name = "x";
    EXPECT_THROW(streaming_pipeline pipe(std::move(cfg)), cpu_snap::config_error);
}

TEST(StreamingPipeline, ConstructionEmitsFileHeaderAndMapsInitAfterFlush)
{
    sink_capture sink;
    {
        streaming_pipeline pipe(make_cfg(sink.as_fn()));
        pipe.finalize();
    }

    // file_header is the first emission; first chunk holds MAPS_INIT
    // (then CHECKPOINT + EOF from finalize).
    ASSERT_GE(sink.chunks.size(), 2U);
    EXPECT_EQ(sink.chunks[0].size(), sizeof(file_header));

    file_header hdr{};
    std::memcpy(&hdr, sink.chunks[0].data(), sizeof(hdr));
    EXPECT_EQ(hdr.magic[0], 'C');
    EXPECT_EQ(hdr.magic[1], 'S');
    EXPECT_EQ(hdr.magic[2], 'N');
    EXPECT_EQ(hdr.magic[3], 'P');
    EXPECT_EQ(hdr.pid, 4242U);

    const auto records = decode_stream(sink);
    ASSERT_GE(records.size(), 3U);
    EXPECT_EQ(records.front().type, record_type::maps_init);
    EXPECT_EQ(records.back().type, record_type::eof);
}

// first sample emits new_nodes + sample

TEST(StreamingPipeline, FirstSampleEmitsThreeNewNodesThenSample)
{
    sink_capture sink;
    {
        streaming_pipeline  pipe(make_cfg(sink.as_fn()));
        const std::uint64_t frames[] = { 0xA, 0xB, 0xC };
        pipe.on_sample(42, 100, std::span<const std::uint64_t>(frames, 3));
        pipe.finalize();
    }

    const auto records = decode_stream(sink);
    // Expected order: MAPS_INIT, NEW_NODE(C), NEW_NODE(B), NEW_NODE(A),
    // SAMPLE, CHECKPOINT, EOF. (Outermost = frames[n-1] = 0xC, becomes
    // root child; innermost = frames[0] = 0xA, becomes leaf.)
    ASSERT_GE(records.size(), 7U);
    EXPECT_EQ(records[0].type, record_type::maps_init);

    const auto nn0 = decode_new_node(records[1]);
    const auto nn1 = decode_new_node(records[2]);
    const auto nn2 = decode_new_node(records[3]);
    EXPECT_EQ(nn0.parent_id, 0U);  // root child
    EXPECT_EQ(nn0.ip, 0xCU);
    EXPECT_EQ(nn1.parent_id, nn0.node_id);
    EXPECT_EQ(nn1.ip, 0xBU);
    EXPECT_EQ(nn2.parent_id, nn1.node_id);
    EXPECT_EQ(nn2.ip, 0xAU);

    const auto s = decode_sample(records[4]);
    EXPECT_EQ(s.timestamp_ns, 100U);
    EXPECT_EQ(s.tid, 42U);
    EXPECT_EQ(s.leaf_node_id, nn2.node_id);

    EXPECT_EQ(records[records.size() - 2].type, record_type::checkpoint);
    EXPECT_EQ(records.back().type, record_type::eof);
}

// steady-state: repeated same stack -> only sample records

TEST(StreamingPipeline, SteadyStateEmitsOnlySamplesAfterFirst)
{
    sink_capture sink;
    {
        streaming_pipeline  pipe(make_cfg(sink.as_fn(), /*ckpt*/ 1000, /*maps*/ 1000));
        const std::uint64_t frames[] = { 0xA, 0xB, 0xC };
        for(std::uint64_t ts = 0; ts < 10000; ++ts)
        {
            pipe.on_sample(42, ts, std::span<const std::uint64_t>(frames, 3));
        }
        EXPECT_EQ(pipe.samples_emitted(), 10000U);
        EXPECT_EQ(pipe.new_nodes_emitted(), 3U);
        // 10000 samples / interval 1000 = 10 periodic + 1 from finalize.
        EXPECT_EQ(pipe.checkpoints_emitted(), 10U);
        pipe.finalize();
        EXPECT_EQ(pipe.checkpoints_emitted(), 11U);
    }

    const auto records = decode_stream(sink);

    std::size_t n_new_node   = 0;
    std::size_t n_sample     = 0;
    std::size_t n_checkpoint = 0;
    std::size_t n_eof        = 0;
    std::size_t n_maps_init  = 0;
    for(const auto& r : records)
    {
        switch(r.type)
        {
            case record_type::new_node: ++n_new_node; break;
            case record_type::sample: ++n_sample; break;
            case record_type::checkpoint: ++n_checkpoint; break;
            case record_type::eof: ++n_eof; break;
            case record_type::maps_init: ++n_maps_init; break;
            default: break;
        }
    }
    EXPECT_EQ(n_new_node, 3U);
    EXPECT_EQ(n_sample, 10000U);
    EXPECT_EQ(n_checkpoint, 11U);
    EXPECT_EQ(n_eof, 1U);
    EXPECT_EQ(n_maps_init, 1U);
}

// multiple tids interleave correctly

TEST(StreamingPipeline, MultipleTidsDistinctLeavesAndTids)
{
    sink_capture sink;
    {
        streaming_pipeline  pipe(make_cfg(sink.as_fn()));
        const std::uint64_t f1[] = { 0x100, 0x200 };
        const std::uint64_t f2[] = { 0x300, 0x400 };
        for(int i = 0; i < 5; ++i)
        {
            pipe.on_sample(1, 1000 + i, std::span<const std::uint64_t>(f1, 2));
            pipe.on_sample(2, 2000 + i, std::span<const std::uint64_t>(f2, 2));
        }
        pipe.finalize();
    }

    const auto records = decode_stream(sink);

    std::uint32_t leaf_tid1 = 0;
    std::uint32_t leaf_tid2 = 0;
    std::size_t   n_tid1    = 0;
    std::size_t   n_tid2    = 0;
    for(const auto& r : records)
    {
        if(r.type != record_type::sample) continue;
        const auto s = decode_sample(r);
        if(s.tid == 1)
        {
            if(leaf_tid1 == 0) leaf_tid1 = s.leaf_node_id;
            EXPECT_EQ(s.leaf_node_id, leaf_tid1);
            ++n_tid1;
        }
        else if(s.tid == 2)
        {
            if(leaf_tid2 == 0) leaf_tid2 = s.leaf_node_id;
            EXPECT_EQ(s.leaf_node_id, leaf_tid2);
            ++n_tid2;
        }
    }
    EXPECT_EQ(n_tid1, 5U);
    EXPECT_EQ(n_tid2, 5U);
    EXPECT_NE(leaf_tid1, 0U);
    EXPECT_NE(leaf_tid2, 0U);
    EXPECT_NE(leaf_tid1, leaf_tid2);
}

// lost samples

TEST(StreamingPipeline, OnLostEmitsLostRecordsAndAggregates)
{
    sink_capture sink;
    {
        streaming_pipeline pipe(make_cfg(sink.as_fn()));
        pipe.on_lost(7);
        pipe.on_lost(13);
        EXPECT_EQ(pipe.lost_total(), 20U);
        pipe.finalize();
    }

    const auto               records = decode_stream(sink);
    std::vector<lost_record> losses;
    for(const auto& r : records)
    {
        if(r.type == record_type::lost) losses.push_back(decode_lost(r));
    }
    ASSERT_EQ(losses.size(), 2U);
    EXPECT_EQ(losses[0].lost_count_delta, 7U);
    EXPECT_EQ(losses[0].tid, 0U);
    EXPECT_EQ(losses[1].lost_count_delta, 13U);
    EXPECT_EQ(losses[1].tid, 0U);
}

TEST(StreamingPipeline, ExternalCountersTrackSamplesAndLostEvenAfterFinalize)
{
    sink_capture             sink;
    std::atomic<std::size_t> total{ 0 };
    std::atomic<std::size_t> lost{ 0 };
    auto                     cfg = make_cfg(sink.as_fn());
    cfg.total_counter            = &total;
    cfg.lost_counter             = &lost;

    streaming_pipeline  pipe(std::move(cfg));
    const std::uint64_t frames[] = { 0xA };
    pipe.on_sample(1, 100, std::span<const std::uint64_t>(frames, 1));
    pipe.on_sample(1, 200, std::span<const std::uint64_t>(frames, 1));
    pipe.on_lost(5);
    pipe.finalize();
    pipe.on_sample(1, 300, std::span<const std::uint64_t>(frames, 1));
    pipe.on_lost(2);

    EXPECT_EQ(total.load(), 3U);
    EXPECT_EQ(lost.load(), 7U);
}

// checkpoint contents

TEST(StreamingPipeline, CheckpointContainsPerTidEntries)
{
    sink_capture sink;
    {
        // checkpoint every 5 samples; 5 on tid 1 + 5 on tid 2 = 10 samples
        // -> 2 periodic checkpoints + finalize's closing one.
        streaming_pipeline  pipe(make_cfg(sink.as_fn(), /*ckpt*/ 5, /*maps*/ 10000));
        const std::uint64_t f1[] = { 0xAAA };
        const std::uint64_t f2[] = { 0xBBB };
        for(int i = 0; i < 5; ++i) {
            pipe.on_sample(1, i, std::span<const std::uint64_t>(f1, 1));
}
        for(int i = 0; i < 5; ++i) {
            pipe.on_sample(2, 100 + i, std::span<const std::uint64_t>(f2, 1));
}
        pipe.finalize();
    }

    const auto records = decode_stream(sink);

    // Find the second periodic checkpoint - it's after both tids have
    // sampled and should list both.
    std::vector<decoded_checkpoint> checkpoints;
    for(const auto& r : records)
    {
        if(r.type == record_type::checkpoint) checkpoints.push_back(decode_checkpoint(r));
    }
    ASSERT_GE(checkpoints.size(), 2U);

    const auto& cp = checkpoints[1];
    ASSERT_EQ(cp.entries.size(), 2U);

    bool saw_tid1 = false;
    bool saw_tid2 = false;
    for(const auto& e : cp.entries)
    {
        if(e.tid == 1U)
        {
            saw_tid1 = true;
            EXPECT_EQ(e.samples, 5U);
            EXPECT_NE(e.leaf_id, 0U);
        }
        else if(e.tid == 2U)
        {
            saw_tid2 = true;
            EXPECT_EQ(e.samples, 5U);
            EXPECT_NE(e.leaf_id, 0U);
        }
    }
    EXPECT_TRUE(saw_tid1);
    EXPECT_TRUE(saw_tid2);
}

// finalize emits EOF and flushes

TEST(StreamingPipeline, FinalizeEmitsEofAndIsIdempotent)
{
    sink_capture sink;
    {
        streaming_pipeline  pipe(make_cfg(sink.as_fn()));
        const std::uint64_t frames[] = { 0xCAFE };
        pipe.on_sample(1, 1, std::span<const std::uint64_t>(frames, 1));
        pipe.finalize();
        const auto bytes_after_first = sink.all().size();

        // Double finalize is a no-op.
        pipe.finalize();
        EXPECT_EQ(sink.all().size(), bytes_after_first);

        // on_sample after finalize is a no-op.
        const auto samples_before = pipe.samples_emitted();
        pipe.on_sample(1, 2, std::span<const std::uint64_t>(frames, 1));
        EXPECT_EQ(pipe.samples_emitted(), samples_before);
    }

    const auto records = decode_stream(sink);
    ASSERT_FALSE(records.empty());
    EXPECT_EQ(records.back().type, record_type::eof);
    const auto eof = decode_eof(records.back());
    EXPECT_EQ(eof.total_samples_emitted, 1U);
}

TEST(StreamingPipeline, DestructorFinalizesWhenNotCalled)
{
    sink_capture sink;
    {
        streaming_pipeline  pipe(make_cfg(sink.as_fn()));
        const std::uint64_t frames[] = { 0x1234 };
        pipe.on_sample(7, 1, std::span<const std::uint64_t>(frames, 1));
    }  // pipe destroyed here; should finalize.

    const auto records = decode_stream(sink);
    ASSERT_FALSE(records.empty());
    EXPECT_EQ(records.back().type, record_type::eof);
}

// trie cap is absorbed by the noexcept hot path

TEST(StreamingPipeline, TrieCapOverflowDoesNotCrash)
{
    sink_capture sink;
    {
        // Cap at 3 nodes. Four distinct single-frame stacks: first three
        // succeed, fourth one throws inside diff(); pipeline catches and
        // increments internal_errors().
        streaming_pipeline  pipe(make_cfg(sink.as_fn(), 1000, 1000, /*max*/ 3));
        const std::uint64_t f1[] = { 0x1 };
        const std::uint64_t f2[] = { 0x2 };
        const std::uint64_t f3[] = { 0x3 };
        const std::uint64_t f4[] = { 0x4 };
        pipe.on_sample(1, 10, std::span<const std::uint64_t>(f1, 1));
        pipe.on_sample(2, 20, std::span<const std::uint64_t>(f2, 1));
        pipe.on_sample(3, 30, std::span<const std::uint64_t>(f3, 1));
        pipe.on_sample(4, 40, std::span<const std::uint64_t>(f4, 1));

        EXPECT_EQ(pipe.samples_emitted(), 3U);
        EXPECT_GE(pipe.internal_errors(), 1U);
        pipe.finalize();
    }
}

// sink throwing does not propagate

TEST(StreamingPipeline, ThrowingSinkDoesNotPropagate)
{
    std::atomic<int> call_count{ 0 };
    sink_fn          throwing = [&](std::span<const std::byte>) {
        const int n = ++call_count;
        if(n % 2 == 0) throw std::runtime_error("simulated sink failure");
    };

    auto run = [&]() {
        streaming_pipeline  pipe(make_cfg(throwing, 1000, 1000, 0, 4096));
        const std::uint64_t frames[] = { 0xAA, 0xBB };
        for(int i = 0; i < 200; ++i)
        {
            pipe.on_sample(static_cast<pid_t>(i % 4), static_cast<std::uint64_t>(i),
                           std::span<const std::uint64_t>(frames, 2));
        }
        pipe.finalize();
        EXPECT_GE(pipe.samples_emitted(), 200U);
    };
    EXPECT_NO_THROW(run());
    EXPECT_GT(call_count.load(), 0);
}

// on_tid_exit drops per-tid counters

TEST(StreamingPipeline, OnTidExitRemovesTidFromSubsequentCheckpoint)
{
    sink_capture sink;
    {
        streaming_pipeline  pipe(make_cfg(sink.as_fn(), /*ckpt*/ 3, /*maps*/ 10000));
        const std::uint64_t f1[] = { 0xAA };
        const std::uint64_t f2[] = { 0xBB };
        pipe.on_sample(10, 1, std::span<const std::uint64_t>(f1, 1));
        pipe.on_sample(20, 2, std::span<const std::uint64_t>(f2, 1));
        pipe.on_sample(10, 3, std::span<const std::uint64_t>(f1, 1));
        // First periodic checkpoint after sample 3 lists both tids.
        pipe.on_tid_exit(20);
        pipe.on_sample(10, 4, std::span<const std::uint64_t>(f1, 1));
        pipe.on_sample(10, 5, std::span<const std::uint64_t>(f1, 1));
        pipe.on_sample(10, 6, std::span<const std::uint64_t>(f1, 1));
        // Second periodic checkpoint after sample 6 lists only tid 10.
        pipe.finalize();
    }

    const auto                      records = decode_stream(sink);
    std::vector<decoded_checkpoint> checkpoints;
    for(const auto& r : records)
    {
        if(r.type == record_type::checkpoint) checkpoints.push_back(decode_checkpoint(r));
    }
    ASSERT_GE(checkpoints.size(), 2U);
    EXPECT_EQ(checkpoints[0].entries.size(), 2U);
    EXPECT_EQ(checkpoints[1].entries.size(), 1U);
    EXPECT_EQ(checkpoints[1].entries[0].tid, 10U);
}

// robustness: lost chunks, trie cap, oversized records

namespace
{

using cpu_snap::sampling::dl_module_watcher;

dl_module_watcher::iterator_fn
make_modules(std::size_t count)
{
    return [count](std::function<void(const dl_module_watcher::injected_module&)> visit) {
        for(std::size_t i = 0; i < count; ++i)
        {
            dl_module_watcher::injected_module mod;
            mod.base_va = 0x10000 * (i + 1);
            mod.size    = 0x1000;
            mod.path    = "/usr/lib/x86_64-linux-gnu/some/long/directory/libmodule_" +
                       std::to_string(i) + ".so";
            visit(mod);
        }
    };
}

streaming_pipeline::config
make_robust_cfg(sink_fn sink, std::size_t trie_max_nodes = 0, std::uint32_t chunk_bytes = 256)
{
    auto cfg = make_cfg(std::move(sink), 1U << 20, 1U << 20, trie_max_nodes, chunk_bytes);
    cfg.module_iterator_for_tests = make_modules(0);
    return cfg;
}

[[nodiscard]] std::vector<std::uint64_t>
last_stack(const cpu_snap::core::sample_reconstructor& rec, pid_t tid)
{
    const auto& samples = rec.samples();
    const auto  it      = samples.find(tid);
    if(it == samples.end() || it->second.empty()) return {};
    const auto& last = it->second.back();
    return { last.frames.begin(), last.frames.end() };
}

}  // namespace

TEST(StreamingPipelineRobustness, SamplesDecodeAgainAfterSinkDropsAChunk)
{
    std::vector<std::byte> bytes;
    std::size_t            calls = 0;
    sink_fn                sink  = [&](std::span<const std::byte> chunk) {
        if(++calls == 2) throw std::runtime_error("sink failure");
        bytes.insert(bytes.end(), chunk.begin(), chunk.end());
    };

    streaming_pipeline pipe(make_robust_cfg(sink));
    for(std::uint64_t i = 0; i < 200; ++i)
    {
        const std::uint64_t frames[] = { 0x1000 + i, 0xA, 0xB };
        pipe.on_sample(100, i, frames);
    }
    pipe.finalize();

    cpu_snap::core::sample_reconstructor rec;
    rec.feed(bytes);

    EXPECT_GE(calls, 2U);
    EXPECT_EQ(last_stack(rec, 100), (std::vector<std::uint64_t>{ 0x1000 + 199, 0xA, 0xB }));
    EXPECT_GE(rec.samples().at(100).size(), 150U);
}

TEST(StreamingPipelineRobustness, TrieCapMidStackStillLetsLaterSamplesDecode)
{
    sink_capture sink;
    streaming_pipeline pipe(make_robust_cfg(sink.as_fn(), 4, 4096));

    const std::uint64_t first[]   = { 0x1, 0xA, 0xB };
    const std::uint64_t capped[]  = { 0x2, 0xC, 0xB };
    const std::uint64_t shorter[] = { 0xC, 0xB };
    pipe.on_sample(1, 10, first);
    pipe.on_sample(2, 20, capped);
    pipe.on_sample(2, 30, shorter);
    pipe.finalize();

    cpu_snap::core::sample_reconstructor rec;
    rec.feed(sink.all());

    EXPECT_EQ(last_stack(rec, 2), (std::vector<std::uint64_t>{ 0xC, 0xB }));
}

TEST(StreamingPipelineRobustness, SnapshotMatchesCounterGetters)
{
    sink_capture sink;
    streaming_pipeline pipe(make_robust_cfg(sink.as_fn(), 2, 4096));

    const std::uint64_t over_cap[] = { 0x1, 0xA, 0xB };
    pipe.on_sample(5, 10, over_cap);
    pipe.finalize();

    const auto snap = pipe.snapshot();
    EXPECT_EQ(snap.dropped_samples, 1U);
    EXPECT_EQ(snap.dropped_samples, pipe.dropped_samples());
    EXPECT_EQ(snap.internal_errors, pipe.internal_errors());
}

TEST(StreamingPipelineRobustness, SampleDroppedByTrieCapIsReportedAsLost)
{
    sink_capture sink;
    streaming_pipeline pipe(make_robust_cfg(sink.as_fn(), 2, 4096));

    const std::uint64_t over_cap[] = { 0x1, 0xA, 0xB };
    pipe.on_sample(5, 10, over_cap);
    pipe.finalize();

    EXPECT_EQ(pipe.dropped_samples(), 1U);

    std::uint64_t lost_for_tid5 = 0;
    std::uint64_t checkpoint_lost = 0;
    for(const auto& rec : decode_stream(sink))
    {
        if(rec.type == record_type::lost)
        {
            const auto lr = decode_lost(rec);
            if(lr.tid == 5) lost_for_tid5 += lr.lost_count_delta;
        }
        if(rec.type == record_type::checkpoint)
        {
            for(const auto& e : decode_checkpoint(rec).entries)
            {
                if(e.tid == 5) checkpoint_lost = e.lost;
            }
        }
    }
    EXPECT_EQ(lost_for_tid5, 1U);
    EXPECT_EQ(checkpoint_lost, 1U);
}

TEST(StreamingPipelineRobustness, LostCountAboveU32IsNotTruncated)
{
    sink_capture sink;
    streaming_pipeline pipe(make_robust_cfg(sink.as_fn(), 0, 4096));

    const std::uint64_t big = (std::uint64_t{ 1 } << 32) + 5;
    pipe.on_lost(big);
    pipe.finalize();

    std::uint64_t total = 0;
    for(const auto& rec : decode_stream(sink))
    {
        if(rec.type == record_type::lost) total += decode_lost(rec).lost_count_delta;
    }
    EXPECT_EQ(total, big);
    EXPECT_EQ(pipe.lost_total(), big);
}

TEST(StreamingPipelineRobustness, DefaultConfigCapsTheTrie)
{
    const streaming_pipeline::config cfg;
    EXPECT_NE(cfg.trie_max_nodes, 0U);
}

TEST(StreamingPipelineRobustness, ManyModulesDoNotOverflowMapsInit)
{
    sink_capture sink;
    auto         cfg       = make_robust_cfg(sink.as_fn(), 0, 1U << 16);
    cfg.module_iterator_for_tests = make_modules(700);

    {
        streaming_pipeline pipe(std::move(cfg));
        pipe.finalize();
        EXPECT_EQ(pipe.internal_errors(), 0U);
    }

    cpu_snap::core::sample_reconstructor rec;
    rec.feed(sink.all());
    for(std::uint64_t i = 0; i < 700; ++i)
    {
        EXPECT_TRUE(rec.modules().resolve(0x10000 * (i + 1) + 8, ~0ULL).has_value()) << i;
    }
}

TEST(StreamingPipelineRobustness, ManyTidsDoNotOverflowCheckpoint)
{
    sink_capture sink;
    streaming_pipeline pipe(make_robust_cfg(sink.as_fn(), 0, 1U << 16));

    const std::uint64_t frames[] = { 0x1 };
    for(pid_t tid = 1; tid <= 3000; ++tid)
    {
        pipe.on_sample(tid, 1, frames);
    }
    pipe.finalize();

    EXPECT_EQ(pipe.internal_errors(), 0U);
    std::size_t entries = 0;
    for(const auto& rec : decode_stream(sink))
    {
        if(rec.type == record_type::checkpoint) entries += decode_checkpoint(rec).entries.size();
    }
    EXPECT_EQ(entries, 3000U);
}

TEST(StreamingPipelineRobustness, PersistentSinkFailureDoesNotReplayTrieEverySample)
{
    std::size_t sink_calls = 0;
    sink_fn     sink       = [&](std::span<const std::byte>) {
        ++sink_calls;
        throw std::runtime_error("down");
    };

    streaming_pipeline pipe(make_robust_cfg(sink));
    for(std::uint64_t i = 0; i < 2000; ++i)
    {
        const std::uint64_t frames[] = { 0x1000 + i, 0xA };
        pipe.on_sample(100, i, frames);
    }

    EXPECT_GT(sink_calls, 0U);
    EXPECT_LT(pipe.new_nodes_emitted(), 100000U);
}

TEST(StreamingPipelineRobustness, CountersAreReadableWhileTheWriterRuns)
{
    sink_fn sink = [](std::span<const std::byte>) { throw std::runtime_error("down"); };
    streaming_pipeline pipe(make_robust_cfg(sink));

    constexpr std::uint64_t k_samples = 5000;
    std::atomic<bool>       done{ false };
    std::uint64_t           last_seen = 0;

    std::thread reader([&] {
        while(!done.load(std::memory_order_acquire))
        {
            const auto emitted = pipe.samples_emitted();
            EXPECT_GE(emitted, last_seen);
            last_seen = emitted;
            (void) pipe.internal_errors();
            (void) pipe.lost_total();
            (void) pipe.checkpoints_emitted();
            (void) pipe.dropped_samples();
            (void) pipe.new_nodes_emitted();
        }
    });

    for(std::uint64_t i = 0; i < k_samples; ++i)
    {
        const std::uint64_t frames[] = { 0x1000 + i, 0xA };
        pipe.on_sample(100, i, frames);
        pipe.on_lost(1);
    }
    done.store(true, std::memory_order_release);
    reader.join();

    EXPECT_EQ(pipe.lost_total(), k_samples);
    EXPECT_EQ(pipe.samples_emitted(), k_samples);
}
