// Integration tests for cpu_snap::sampling::sampler in streaming mode.
//
// Drives the recorder with a non-null sink, captures bytes
// into an in-memory buffer, and verifies the wire stream framing
// (file_header / chunks / EOF). Tests requiring kernel perf permissions
// GTEST_SKIP when perf_event_paranoid > 1.

#include <cstdio>
#include <cstddef>
#include <functional>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "cpu_snap/error.hpp"
#include "cpu_snap/core/reader.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/types.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"
#include "cpu_snap/sampling/unwind_mode.hpp"

namespace
{

[[gnu::noinline]] double
busy_inner(double s, std::uint64_t n) noexcept
{
    double acc = s;
    for(std::uint64_t i = 0; i < n; ++i) {
        acc = acc * 1.0000001 + (i & 0xFFFF);
}
    return acc;
}

[[gnu::noinline]] double
busy_outer(double s, std::uint64_t n) noexcept
{
    volatile double r = busy_inner(s + 1.0, n);
    return r;
}

void
spin_worker(std::atomic<bool>& stop)
{
    double acc = 0.0;
    while(!stop.load(std::memory_order_relaxed)) {
        acc += busy_outer(acc, 100'000);
}
    if(acc == 12345.6789) std::printf("unreachable\n");
}

bool
perf_open_available()
{
    struct perf_event_attr a
    {};
    a.size           = sizeof(a);
    a.type           = PERF_TYPE_SOFTWARE;
    a.config         = PERF_COUNT_SW_CPU_CLOCK;
    a.disabled       = 1;
    a.exclude_kernel = 1;
    a.exclude_hv     = 1;
    const int fd     = static_cast<int>(
        ::syscall(SYS_perf_event_open, &a, 0, -1, -1, PERF_FLAG_FD_CLOEXEC));
    if(fd < 0) return false;
    ::close(fd);
    return true;
}

/**
 * Captures every span the sink receives, copying out the bytes.
 * Mutex-protected because the recorder's worker thread emits chunks
 * from a background thread; the main test thread reads after stop().
 */
struct sink_capture
{
    std::mutex                          mu;
    std::vector<std::vector<std::byte>> chunks;
    std::atomic<std::uint64_t>          call_count{ 0 };

    cpu_snap::core::sink_fn as_fn()
    {
        return [this](std::span<const std::byte> bytes) {
            std::lock_guard<std::mutex> const lk(mu);
            chunks.emplace_back(bytes.begin(), bytes.end());
            call_count.fetch_add(1, std::memory_order_relaxed);
        };
    }

    [[nodiscard]] std::size_t total_bytes() const
    {
        std::size_t n = 0;
        for(const auto& c : chunks) {
            n += c.size();
}
        return n;
    }
};

// Wire-format chunk-stream parser. Lifted from
// tests/internal/test_streaming_pipeline.cpp (decode_stream helpers)
// to keep this test self-contained.
struct decoded_record
{
    cpu_snap::core::record_type  type = {};
    std::vector<std::byte> body;
};

[[nodiscard]] std::vector<decoded_record>
decode_records(const sink_capture& sink)
{
    using cpu_snap::core::chunk_header;
    using cpu_snap::core::file_header;
    using cpu_snap::core::record_type;

    std::vector<decoded_record> out;
    if(sink.chunks.empty()) return out;

    EXPECT_EQ(sink.chunks[0].size(), sizeof(file_header));

    for(std::size_t ci = 1; ci < sink.chunks.size(); ++ci)
    {
        const auto& bytes = sink.chunks[ci];
        if(bytes.size() < sizeof(chunk_header)) continue;

        chunk_header ch{};
        std::memcpy(&ch, bytes.data(), sizeof(ch));
        EXPECT_EQ(ch.magic[0], 'C');
        EXPECT_EQ(ch.magic[1], 'H');

        const std::byte*  body_begin = bytes.data() + sizeof(chunk_header);
        const std::size_t body_size  = ch.records_byte_count;
        const std::byte*  body_end   = body_begin + body_size;

        const std::byte* p = body_begin;
        while(p < body_end)
        {
            std::uint8_t  type_byte = 0;
            std::uint8_t  reserved  = 0;
            std::uint16_t body_sz   = 0;
            std::memcpy(&type_byte, p, 1);
            std::memcpy(&reserved, p + 1, 1);
            std::memcpy(&body_sz, p + 2, 2);

            decoded_record rec;
            rec.type = static_cast<record_type>(type_byte);
            rec.body.assign(p + 4, p + 4 + body_sz);
            out.push_back(std::move(rec));

            const std::uint32_t consumed = 4U + body_sz;
            const std::uint32_t padded   = (consumed + 3U) & ~3U;
            p += padded;
        }
    }
    return out;
}

}  // namespace

// Construction validation. No perf permissions required.

TEST(PerfSamplerStreaming, CtorAcceptsStreamingPlusDwarf)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.unwind     = cpu_snap::sampling::unwind_mode::dwarf;
    cfg.streaming.sink  = [](std::span<const std::byte>) {};
    EXPECT_NO_THROW(cpu_snap::sampling::sampler{ std::move(cfg) });
}

TEST(PerfSamplerStreaming, CtorAcceptsStreamingPlusFramePointer)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.unwind     = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink  = [](std::span<const std::byte>) {};
    EXPECT_NO_THROW(cpu_snap::sampling::sampler{ std::move(cfg) });
}

TEST(PerfSamplerStreaming, CtorAcceptsStreamingPlusAutoBest)
{
    // auto_best resolves to dwarf, which streaming accepts.
    cpu_snap::sampling::sampler::config cfg;
    cfg.streaming.sink  = [](std::span<const std::byte>) {};
    EXPECT_NO_THROW(cpu_snap::sampling::sampler{ std::move(cfg) });
}

// The sink is mandatory: construction without one is a config error.
// No perf permissions required.

TEST(PerfSamplerStreaming, MissingSinkThrowsAtConstruction)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.kernel_pages_per_tid = 1;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    EXPECT_THROW(cpu_snap::sampling::sampler{ std::move(cfg) }, cpu_snap::config_error);
}

// Integration tests. Need perf_event_paranoid <= 1.

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest fixture must be CamelCase to satisfy google-readability-avoid-underscore-in-googletest-name
class PerfSamplerStreamingIntegrationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";
    }
};

TEST_F(PerfSamplerStreamingIntegrationTest, StreamingSinkReceivesBytesAndEof)
{
    sink_capture sink;

    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    {
        cpu_snap::sampling::sampler::config cfg;
        cfg.perf.sample_freq_hz            = 500;
        cfg.perf.callchain_depth           = 16;
        cfg.perf.kernel_pages_per_tid      = 4;
        cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
        cfg.streaming.sink                 = sink.as_fn();
        cfg.streaming.chunk_bytes          = 1U << 16;  // 64 KiB so flushes trigger
        cfg.streaming.checkpoint_interval  = 1000;
        cfg.streaming.maps_poll_interval   = 1000;
        cfg.streaming.process_name         = "sampler_test";

        cpu_snap::sampling::sampler s(std::move(cfg));
        s.start();
        std::this_thread::sleep_for(std::chrono::milliseconds{ 200 });
        s.stop();

        EXPECT_GT(s.total_samples(), 0U);
        EXPECT_GT(s.tracked_threads(), 0U);
    }

    stop.store(true);
    worker.join();

    EXPECT_GT(sink.call_count.load(), 0U);
    ASSERT_GE(sink.chunks.size(), 1U);
    EXPECT_EQ(sink.chunks.front().size(), sizeof(cpu_snap::core::file_header));

    cpu_snap::core::file_header hdr{};
    std::memcpy(&hdr, sink.chunks.front().data(), sizeof(hdr));
    EXPECT_EQ(hdr.magic[0], 'C');
    EXPECT_EQ(hdr.magic[1], 'S');
    EXPECT_EQ(hdr.magic[2], 'N');
    EXPECT_EQ(hdr.magic[3], 'P');
    EXPECT_EQ(hdr.pid, static_cast<std::uint64_t>(::getpid()));
    EXPECT_EQ(hdr.flags, 0x2U) << "frame-pointer capture must set the FP flag bit";

    const auto records = decode_records(sink);
    ASSERT_FALSE(records.empty());
    EXPECT_EQ(records.back().type, cpu_snap::core::record_type::eof);
}

TEST_F(PerfSamplerStreamingIntegrationTest, AutoBestResolvesToDwarfForPerfBackend)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.backend        = cpu_snap::sampling::backend_kind::perf;
    cfg.streaming.sink = [](std::span<const std::byte>) {};
    cpu_snap::sampling::sampler s(std::move(cfg));
    EXPECT_EQ(s.resolved_unwind_mode(), cpu_snap::sampling::unwind_mode::dwarf);
}

TEST_F(PerfSamplerStreamingIntegrationTest, DroppedSamplesStayReadableAfterStop)
{
    sink_capture sink;

    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.sample_freq_hz       = 500;
    cfg.perf.callchain_depth      = 16;
    cfg.perf.kernel_pages_per_tid = 4;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink            = sink.as_fn();
    cfg.streaming.trie_max_nodes  = 1;

    cpu_snap::sampling::sampler s(std::move(cfg));
    s.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 200 });
    s.stop();

    stop.store(true);
    worker.join();

    ASSERT_GT(s.total_samples(), 0U);
    EXPECT_GT(s.streaming_dropped_samples(), 0U);
}

TEST_F(PerfSamplerStreamingIntegrationTest, FinalizeEmitsEofAfterStop)
{
    sink_capture sink;

    {
        cpu_snap::sampling::sampler::config cfg;
        cfg.perf.sample_freq_hz       = 500;
        cfg.perf.kernel_pages_per_tid = 1;
        cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
        cfg.streaming.sink            = sink.as_fn();
        cfg.streaming.chunk_bytes     = 1U << 16;
        cfg.streaming.process_name    = "eof_test";

        cpu_snap::sampling::sampler s(std::move(cfg));
        s.start();
        std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });
        s.stop();
    }

    const auto records = decode_records(sink);
    ASSERT_FALSE(records.empty());
    EXPECT_EQ(records.back().type, cpu_snap::core::record_type::eof);
}

TEST_F(PerfSamplerStreamingIntegrationTest, DwarfUnwindSetsDwarfFlagInFileHeader)
{
    sink_capture sink;

    {
        cpu_snap::sampling::sampler::config cfg;
        cfg.perf.sample_freq_hz       = 500;
        cfg.perf.kernel_pages_per_tid = 1;
        cfg.perf.dwarf_stack_bytes    = 2048;
        cfg.perf.unwind               = cpu_snap::sampling::unwind_mode::dwarf;
        cfg.streaming.sink            = sink.as_fn();
        cfg.streaming.chunk_bytes     = 1U << 16;
        cfg.streaming.process_name    = "dwarf_flags_test";

        cpu_snap::sampling::sampler s(std::move(cfg));
        ASSERT_EQ(s.resolved_unwind_mode(), cpu_snap::sampling::unwind_mode::dwarf);
        s.start();
        std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });
        s.stop();
    }

    ASSERT_GE(sink.chunks.size(), 1U);
    EXPECT_EQ(sink.chunks.front().size(), sizeof(cpu_snap::core::file_header));

    cpu_snap::core::file_header hdr{};
    std::memcpy(&hdr, sink.chunks.front().data(), sizeof(hdr));
    EXPECT_EQ(hdr.flags, 0x4U) << "DWARF capture must set the DWARF flag bit, not the FP bit";
}

TEST_F(PerfSamplerStreamingIntegrationTest, DwarfUnwindStreamsResolvableCallChain)
{
    sink_capture sink;

    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    {
        cpu_snap::sampling::sampler::config cfg;
        cfg.perf.sample_freq_hz       = 500;
        cfg.perf.callchain_depth      = 16;
        cfg.perf.kernel_pages_per_tid = 4;
        cfg.perf.dwarf_stack_bytes    = 8192;
        cfg.perf.unwind               = cpu_snap::sampling::unwind_mode::dwarf;
        cfg.streaming.sink            = sink.as_fn();
        cfg.streaming.chunk_bytes     = 1U << 16;
        cfg.streaming.process_name    = "dwarf_streaming_test";

        cpu_snap::sampling::sampler s(std::move(cfg));
        ASSERT_EQ(s.resolved_unwind_mode(), cpu_snap::sampling::unwind_mode::dwarf);
        s.start();
        std::this_thread::sleep_for(std::chrono::milliseconds{ 300 });
        s.stop();
        EXPECT_GT(s.total_samples(), 0U);
    }

    stop.store(true);
    worker.join();

    cpu_snap::core::reader reader;
    std::vector<std::vector<std::uint64_t>> stacks;
    reader.on_event([&](const cpu_snap::core::record_variant& var) {
        if(cpu_snap::core::type_of(var) != cpu_snap::core::record_type::sample) return;
        const auto& sample = std::get<cpu_snap::core::sample_record>(var);
        if(sample.leaf_node_id == 0) return;
        auto ips = reader.resolve_stack(sample.leaf_node_id);
        if(!ips.empty()) stacks.push_back(std::move(ips));
    });
    for(const auto& chunk : sink.chunks)
    {
        reader.feed(std::span<const std::byte>(chunk.data(), chunk.size()));
    }
    ASSERT_FALSE(stacks.empty());

    std::vector<std::uint64_t> all_ips;
    for(const auto& stack : stacks)
    {
        all_ips.insert(all_ips.end(), stack.begin(), stack.end());
    }

    cpu_snap::processing::symbols::symbolizer sym(
        cpu_snap::processing::symbols::symbolizer_config{ .include_binary_name = false });
    sym.preload(all_ips);

    bool found_ordered_chain = false;
    for(const auto& stack : stacks)
    {
        int inner_idx = -1;
        int outer_idx = -1;
        int worker_idx = -1;
        for(std::size_t i = 0; i < stack.size(); ++i)
        {
            const std::string_view name = sym.resolve(stack[i]);
            if(inner_idx < 0 && name.find("busy_inner") != std::string_view::npos)
                inner_idx = static_cast<int>(i);
            if(outer_idx < 0 && name.find("busy_outer") != std::string_view::npos)
                outer_idx = static_cast<int>(i);
            if(worker_idx < 0 && name.find("spin_worker") != std::string_view::npos)
                worker_idx = static_cast<int>(i);
        }
        if(inner_idx >= 0 && outer_idx > inner_idx && worker_idx > outer_idx)
        {
            found_ordered_chain = true;
            break;
        }
    }
    EXPECT_TRUE(found_ordered_chain)
        << "expected at least one sample with innermost-first order "
           "busy_inner -> busy_outer -> spin_worker";
}

TEST_F(PerfSamplerStreamingIntegrationTest, PauseResumeKeepsStreamWellformed)
{
    sink_capture sink;

    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    {
        cpu_snap::sampling::sampler::config cfg;
        cfg.perf.sample_freq_hz       = 500;
        cfg.perf.kernel_pages_per_tid = 4;
        cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
        cfg.streaming.sink            = sink.as_fn();
        cfg.streaming.chunk_bytes     = 1U << 16;
        cfg.streaming.process_name    = "pause_test";

        cpu_snap::sampling::sampler s(std::move(cfg));
        s.start();
        std::this_thread::sleep_for(std::chrono::milliseconds{ 80 });
        s.pause();
        EXPECT_TRUE(s.is_paused());
        std::this_thread::sleep_for(std::chrono::milliseconds{ 80 });
        s.resume();
        EXPECT_FALSE(s.is_paused());
        std::this_thread::sleep_for(std::chrono::milliseconds{ 80 });
        s.stop();
    }

    stop.store(true);
    worker.join();

    const auto records = decode_records(sink);
    ASSERT_FALSE(records.empty());
    EXPECT_EQ(records.back().type, cpu_snap::core::record_type::eof);
}
