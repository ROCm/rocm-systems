// Unit + integration tests for the online cpu_snap::sampling::sampler
// (raw recorder). Tests for the offline sample_postprocessor are in
// tests/symbols/test_sample_postprocessor.cpp.

#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <thread>

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#include "cpu_snap/error.hpp"
#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "cpu_snap/sampling/sampler.hpp"

namespace
{
constexpr std::size_t k_test_sink_max_bytes = 64U * 1024U * 1024U;
}  // namespace

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
    return busy_inner(s + 1.0, n);
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

}  // namespace

// Config / construction validation. No perf permissions required.

TEST(PerfSamplerConfigTest, CtorRejectsZeroCallchainDepth)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.callchain_depth = 0;
    cfg.streaming.sink       = [](std::span<const std::byte>) {};
    EXPECT_THROW(cpu_snap::sampling::sampler{ std::move(cfg) }, cpu_snap::config_error);
}

TEST(PerfSamplerConfigTest, CtorRejectsCallchainOverMax)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.callchain_depth = cpu_snap::core::k_max_callchain_depth + 1;
    cfg.streaming.sink       = [](std::span<const std::byte>) {};
    EXPECT_THROW(cpu_snap::sampling::sampler{ std::move(cfg) }, cpu_snap::config_error);
}

TEST(PerfSamplerConfigTest, CtorAcceptsCallchainBeyondFormerFixedCap)
{
    constexpr std::uint32_t k_former_cap = 128;

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.callchain_depth = 4 * k_former_cap;
    cfg.streaming.sink       = [](std::span<const std::byte>) {};
    EXPECT_NO_THROW(cpu_snap::sampling::sampler{ std::move(cfg) });
}

TEST(PerfSamplerConfigTest, CtorRejectsNonPowerOfTwoPages)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.kernel_pages_per_tid = 7;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink            = [](std::span<const std::byte>) {};
    EXPECT_THROW(cpu_snap::sampling::sampler{ std::move(cfg) }, cpu_snap::config_error);
}

TEST(PerfSamplerConfigTest, CtorAcceptsZeroPagesAsAutomaticSizing)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.kernel_pages_per_tid = 0;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink            = [](std::span<const std::byte>) {};
    EXPECT_NO_THROW(cpu_snap::sampling::sampler{ std::move(cfg) });
}

TEST(PerfSamplerConfigTest, CtorRejectsMissingSink)
{
    cpu_snap::sampling::sampler::config cfg;
    EXPECT_THROW(cpu_snap::sampling::sampler{ std::move(cfg) }, cpu_snap::config_error);
}

TEST(PerfSamplerConfigTest, CtorAcceptsMinimalValidConfig)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.callchain_depth      = 1;
    cfg.perf.kernel_pages_per_tid = 1;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink            = [](std::span<const std::byte>) {};
    EXPECT_NO_THROW(cpu_snap::sampling::sampler{ std::move(cfg) });
}

TEST(PerfSamplerConfigTest, StreamingInternalErrorsZeroByDefault)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.streaming.sink = [](std::span<const std::byte>) {};
    const cpu_snap::sampling::sampler smp{ std::move(cfg) };
    EXPECT_EQ(smp.streaming_internal_errors(), 0U);
}

// Integration tests. Need perf_event_paranoid <= 1.

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest fixture must be CamelCase to satisfy google-readability-avoid-underscore-in-googletest-name
class PerfSamplerIntegrationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";
    }
};

TEST_F(PerfSamplerIntegrationTest, RecordsSamplesForBusyThread)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.sample_freq_hz       = 500;
    cfg.perf.callchain_depth      = 16;
    cfg.perf.kernel_pages_per_tid = 4;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink            = std::ref(sink);

    cpu_snap::sampling::sampler s(std::move(cfg));
    s.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 600 });
    s.stop();

    stop.store(true);
    worker.join();

    EXPECT_GT(s.total_samples(), 0U);

    cpu_snap::core::sample_reconstructor reconstructor;
    reconstructor.feed(sink.bytes());
    std::size_t stored = 0;
    for(const auto& [tid, vec] : reconstructor.samples()) {
        stored += vec.size();
}
    EXPECT_GT(stored, 0U);
    EXPECT_LE(stored, s.total_samples());
}

TEST_F(PerfSamplerIntegrationTest, DoubleStartIsNoop)
{
    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.kernel_pages_per_tid = 1;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink            = std::ref(sink);
    cpu_snap::sampling::sampler s(std::move(cfg));
    s.start();
    EXPECT_NO_THROW(s.start());
    s.stop();
}

TEST_F(PerfSamplerIntegrationTest, StopWithoutStartIsNoop)
{
    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.kernel_pages_per_tid = 1;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink            = std::ref(sink);
    cpu_snap::sampling::sampler s(std::move(cfg));
    EXPECT_NO_THROW(s.stop());
}

TEST_F(PerfSamplerIntegrationTest, DtorStopsRunningSampler)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));
    {
        cpu_snap::core::memory_sink sink(
            cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });

        cpu_snap::sampling::sampler::config cfg;
        cfg.perf.kernel_pages_per_tid = 1;
        cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
        cfg.streaming.sink            = std::ref(sink);
        cpu_snap::sampling::sampler s(std::move(cfg));
        s.start();
        std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });
    }
    stop.store(true);
    worker.join();
    SUCCEED();
}

TEST_F(PerfSamplerIntegrationTest, WatchNewThreadsRecordsPostStartWorker)
{
    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.sample_freq_hz       = 500;
    cfg.perf.kernel_pages_per_tid = 4;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.threads.rescan_period     = std::chrono::milliseconds{ 50 };
    cfg.threads.watch_new_threads = true;
    cfg.streaming.sink            = std::ref(sink);

    cpu_snap::sampling::sampler s(std::move(cfg));
    s.start();

    std::atomic<bool> stop{ false };
    std::thread       late_worker(spin_worker, std::ref(stop));

    std::this_thread::sleep_for(std::chrono::milliseconds{ 500 });
    s.stop();
    stop.store(true);
    late_worker.join();

    cpu_snap::core::sample_reconstructor reconstructor;
    reconstructor.feed(sink.bytes());

    bool any_sample = false;
    for(const auto& [tid, vec] : reconstructor.samples())
    {
        if(!vec.empty())
        {
            any_sample = true;
            break;
        }
    }
    EXPECT_TRUE(any_sample);
}

TEST_F(PerfSamplerIntegrationTest, CallchainDepthCapsFramesPerSample)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });

    constexpr std::uint32_t             cap = 3;
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.sample_freq_hz       = 500;
    cfg.perf.callchain_depth      = cap;
    cfg.perf.kernel_pages_per_tid = 4;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink            = std::ref(sink);

    cpu_snap::sampling::sampler s(std::move(cfg));
    s.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 400 });
    s.stop();

    stop.store(true);
    worker.join();

    cpu_snap::core::sample_reconstructor reconstructor;
    reconstructor.feed(sink.bytes());

    std::size_t total   = 0;
    std::size_t max_obs = 0;
    for(const auto& [tid, vec] : reconstructor.samples())
    {
        total += vec.size();
        for(const auto& rs : vec) {
            max_obs = std::max<std::size_t>(max_obs, rs.frames.size());
}
    }
    EXPECT_GT(total, 0U);
    EXPECT_LE(max_obs, cap);
}

TEST_F(PerfSamplerIntegrationTest, RestartAfterStopRecordsSamplesAgain)
{
    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });

    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.sample_freq_hz       = 500;
    cfg.perf.kernel_pages_per_tid = 4;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink            = std::ref(sink);
    cpu_snap::sampling::sampler s(std::move(cfg));

    s.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 200 });
    s.stop();
    const auto first_total = s.total_samples();
    ASSERT_GT(first_total, 0U);

    EXPECT_NO_THROW(s.start());
    std::this_thread::sleep_for(std::chrono::milliseconds{ 200 });
    s.stop();
    stop.store(true);
    worker.join();

    EXPECT_GT(s.total_samples(), first_total);
}

TEST(PerfSamplerConfigTest, DwarfStackBytesDefaultsTo32KiB)
{
    EXPECT_EQ(cpu_snap::sampling::sampler::perf_attr_options{}.dwarf_stack_bytes, 32768U);
}

TEST(PerfSamplerConfigTest, CtorAcceptsDwarfWithDefaultStackAndAutoRing)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.unwind    = cpu_snap::sampling::unwind_mode::dwarf;
    cfg.streaming.sink = [](std::span<const std::byte>) {};
    EXPECT_NO_THROW(cpu_snap::sampling::sampler{ std::move(cfg) });
}

TEST(PerfSamplerConfigTest, CtorRejectsDwarfRingSmallerThanOneRecord)
{
    cpu_snap::sampling::sampler::config cfg;
    if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";
    cfg.backend = cpu_snap::sampling::backend_kind::perf;
    cfg.perf.unwind               = cpu_snap::sampling::unwind_mode::dwarf;
    cfg.perf.dwarf_stack_bytes    = 32768;
    cfg.perf.kernel_pages_per_tid = 8;
    cfg.streaming.sink            = [](std::span<const std::byte>) {};
    try
    {
        cpu_snap::sampling::sampler{ std::move(cfg) };
        FAIL() << "expected cpu_snap::config_error";
    } catch(const cpu_snap::config_error& e)
    {
        EXPECT_THAT(e.what(), ::testing::HasSubstr("kernel_pages_per_tid"));
    }
}

TEST(PerfSamplerConfigTest, CtorRejectsDwarfStackAboveKernelLimit)
{
    cpu_snap::sampling::sampler::config cfg;
    if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";
    cfg.backend = cpu_snap::sampling::backend_kind::perf;
    cfg.perf.unwind               = cpu_snap::sampling::unwind_mode::dwarf;
    cfg.perf.dwarf_stack_bytes    = 65536;
    cfg.streaming.sink            = [](std::span<const std::byte>) {};
    EXPECT_THROW(cpu_snap::sampling::sampler{ std::move(cfg) }, cpu_snap::config_error);
}

TEST(PerfSamplerConfigTest, CtorRejectsZeroDwarfStackBytes)
{
    cpu_snap::sampling::sampler::config cfg;
    if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";
    cfg.backend = cpu_snap::sampling::backend_kind::perf;
    cfg.perf.unwind            = cpu_snap::sampling::unwind_mode::dwarf;
    cfg.perf.dwarf_stack_bytes = 0;
    cfg.streaming.sink         = [](std::span<const std::byte>) {};
    EXPECT_THROW(cpu_snap::sampling::sampler{ std::move(cfg) }, cpu_snap::config_error);
}

TEST(SamplerConfigTest, TrieIsCappedByDefault)
{
    const cpu_snap::sampling::sampler::config cfg;
    EXPECT_EQ(cfg.streaming.trie_max_nodes, cpu_snap::sampling::internal::k_default_trie_max_nodes);
    EXPECT_GT(cfg.streaming.trie_max_nodes, 0U);
}
