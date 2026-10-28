// Tests for sampler::config::clock: one clock for every timestamp of a stream.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "cpu_snap/core/chunk_framer.hpp"
#include "cpu_snap/core/reader.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/sampling/clock_kind.hpp"
#include "cpu_snap/sampling/logger.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/sampling/time.hpp"
#include "support/stream_capture.hpp"

namespace
{

using namespace std::chrono_literals;
using cpu_snap::core::stream_clock;
using cpu_snap::sampling::backend_kind;
using cpu_snap::sampling::clock_kind;
using cpu_snap::sampling::sampler;
namespace utils = cpu_snap::sampling::utils;

constexpr clock_kind k_all_clocks[] = { clock_kind::monotonic, clock_kind::boottime,
                                        clock_kind::wallclock };

std::uint64_t
read_clock(clock_kind kind)
{
    timespec ts{};
    ::clock_gettime(utils::clock_id_of(kind), &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

stream_clock
expected_stream_clock(clock_kind kind)
{
    switch(kind)
    {
        case clock_kind::boottime: return stream_clock::boottime;
        case clock_kind::wallclock: return stream_clock::wallclock;
        case clock_kind::monotonic: break;
    }
    return stream_clock::monotonic;
}

bool
perf_open_available()
{
    perf_event_attr a{};
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

struct stream_facts
{
    stream_clock               clock = stream_clock::monotonic;
    std::vector<std::uint64_t> sample_ts;
    std::vector<std::uint64_t> other_ts;
    std::size_t                eofs      = 0;
    std::size_t                maps_init = 0;
    std::size_t                parse_errors = 0;
};

struct capture_result
{
    stream_facts  facts;
    std::uint64_t t0 = 0;
    std::uint64_t t1 = 0;
    std::vector<std::pair<std::uint64_t, std::string>> logs;
};

void
spin(std::atomic<bool>& stop)
{
    volatile double acc = 0.0;
    while(!stop.load(std::memory_order_relaxed))
    {
        for(int i = 0; i < 10'000; ++i) acc = acc + 1.0;
    }
}

stream_facts
parse_stream(const std::vector<std::byte>& bytes)
{
    stream_facts      facts;
    cpu_snap::core::reader r;
    r.on_event([&](const cpu_snap::core::record_variant& rec) {
        std::visit(
            [&](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr(std::is_same_v<T, cpu_snap::core::sample_record>)
                {
                    facts.sample_ts.push_back(v.timestamp_ns);
                }
                else if constexpr(std::is_same_v<T, cpu_snap::core::maps_init_record> ||
                                  std::is_same_v<T, cpu_snap::core::maps_delta_record> ||
                                  std::is_same_v<T, cpu_snap::core::checkpoint_record> ||
                                  std::is_same_v<T, cpu_snap::core::lost_record> ||
                                  std::is_same_v<T, cpu_snap::core::eof_record>)
                {
                    facts.other_ts.push_back(v.wall_ns);
                    if constexpr(std::is_same_v<T, cpu_snap::core::eof_record>) ++facts.eofs;
                    if constexpr(std::is_same_v<T, cpu_snap::core::maps_init_record>)
                        ++facts.maps_init;
                }
            },
            rec);
    });
    r.feed(std::span<const std::byte>(bytes.data(), bytes.size()));
    facts.clock        = cpu_snap::core::stream_clock_of(r.header());
    facts.parse_errors = r.counters().parse_errors;
    return facts;
}

capture_result
run_capture(sampler::config cfg)
{
    std::mutex             mtx;
    std::vector<std::byte> bytes;
    capture_result         out;
    cfg.streaming.sink = [&](std::span<const std::byte> chunk) {
        const std::lock_guard<std::mutex> lk(mtx);
        bytes.insert(bytes.end(), chunk.begin(), chunk.end());
    };
    cfg.logger = [&](const cpu_snap::sampling::log_record& r) {
        const std::lock_guard<std::mutex> lk(mtx);
        out.logs.emplace_back(r.timestamp_ns, std::string(r.message));
    };
    cfg.threads.rescan_period = 20ms;

    std::atomic<bool> stop{ false };
    std::thread       worker(spin, std::ref(stop));
    out.t0 = read_clock(cfg.clock);
    {
        const clock_kind clk = cfg.clock;
        sampler s(std::move(cfg));
        s.start();
        std::this_thread::sleep_for(300ms);
        s.stop();
        out.t1 = read_clock(clk);
    }
    stop.store(true);
    worker.join();

    const std::lock_guard<std::mutex> lk(mtx);
    out.facts = parse_stream(bytes);
    return out;
}

void
expect_one_domain(const capture_result& res, clock_kind kind)
{
    EXPECT_EQ(res.facts.parse_errors, 0U);
    EXPECT_EQ(res.facts.clock, expected_stream_clock(kind));
    EXPECT_GE(res.facts.maps_init, 1U);
    EXPECT_EQ(res.facts.eofs, 1U);
    ASSERT_FALSE(res.facts.sample_ts.empty());
    for(const auto ts : res.facts.sample_ts)
    {
        EXPECT_GE(ts, res.t0);
        EXPECT_LE(ts, res.t1);
    }
    for(const auto ts : res.facts.other_ts)
    {
        EXPECT_GE(ts, res.t0);
        EXPECT_LE(ts, res.t1);
    }
}

TEST(clock_option, clock_id_of_maps_each_kind)
{
    EXPECT_EQ(utils::clock_id_of(clock_kind::monotonic), CLOCK_MONOTONIC);
    EXPECT_EQ(utils::clock_id_of(clock_kind::boottime), CLOCK_BOOTTIME);
    EXPECT_EQ(utils::clock_id_of(clock_kind::wallclock), CLOCK_REALTIME);
}

TEST(clock_option, clock_now_ns_lies_between_two_readings_of_the_same_clock)
{
    for(const auto kind : k_all_clocks)
    {
        const auto before = read_clock(kind);
        const auto now    = utils::clock_now_ns(kind);
        const auto after  = read_clock(kind);
        EXPECT_GE(now, before);
        EXPECT_LE(now, after);
    }
}

TEST(clock_option, default_config_is_monotonic)
{
    EXPECT_EQ(sampler::config{}.clock, clock_kind::monotonic);
}

TEST(clock_option, signal_backend_stamps_every_record_with_the_configured_clock)
{
    for(const auto kind : k_all_clocks)
    {
        SCOPED_TRACE(static_cast<int>(kind));
        sampler::config cfg;
        cfg.backend = backend_kind::signal;
        cfg.clock   = kind;
        expect_one_domain(run_capture(std::move(cfg)), kind);
    }
}

TEST(clock_option, signal_backend_default_config_reads_back_monotonic)
{
    sampler::config cfg;
    cfg.backend = backend_kind::signal;
    const auto res = run_capture(std::move(cfg));
    EXPECT_EQ(res.facts.clock, stream_clock::monotonic);
    expect_one_domain(res, clock_kind::monotonic);
}

TEST(clock_option, perf_backend_stamps_every_record_with_the_configured_clock)
{
    if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";
    for(const auto kind : k_all_clocks)
    {
        SCOPED_TRACE(static_cast<int>(kind));
        sampler::config cfg;
        cfg.backend                = backend_kind::perf;
        cfg.perf.sample_freq_hz    = 500;
        cfg.perf.kernel_pages_per_tid = 4;
        cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
        cfg.clock                  = kind;
        expect_one_domain(run_capture(std::move(cfg)), kind);
    }
}

TEST(clock_option, log_records_carry_timestamps_of_the_configured_clock)
{
    for(const auto kind : k_all_clocks)
    {
        SCOPED_TRACE(static_cast<int>(kind));
        sampler::config cfg;
        cfg.backend = backend_kind::signal;
        cfg.clock   = kind;
        const auto res = run_capture(std::move(cfg));
        ASSERT_GE(res.logs.size(), 3U);
        for(const auto& [ts, message] : res.logs)
        {
            EXPECT_GE(ts, res.t0) << message;
            EXPECT_LE(ts, res.t1) << message;
        }
    }
}

std::vector<std::byte>
header_with_flags(std::uint32_t flags)
{
    cpu_snap::test_support::byte_buffer buf;
    auto cfg = cpu_snap::test_support::make_framer_cfg(4096);
    cpu_snap::core::chunk_framer fr(cfg, buf.as_sink());
    fr.start();
    constexpr std::size_t k_flags_offset = offsetof(cpu_snap::core::file_header, flags);
    std::memcpy(buf.bytes.data() + k_flags_offset, &flags, sizeof(flags));
    return buf.bytes;
}

stream_clock
parsed_clock_for_flags(std::uint32_t flags, bool& fatal)
{
    const auto bytes = header_with_flags(flags);
    cpu_snap::core::reader r;
    r.feed(std::span<const std::byte>(bytes.data(), bytes.size()));
    fatal = r.is_fatal();
    return cpu_snap::core::stream_clock_of(r.header());
}

TEST(clock_option, reader_decodes_clock_bits_of_the_header_flags)
{
    bool fatal = true;
    EXPECT_EQ(parsed_clock_for_flags(0x2, fatal), stream_clock::monotonic);
    EXPECT_FALSE(fatal);
    EXPECT_EQ(parsed_clock_for_flags(0x2 | (1U << 3), fatal), stream_clock::boottime);
    EXPECT_FALSE(fatal);
    EXPECT_EQ(parsed_clock_for_flags(0x4 | (2U << 3), fatal), stream_clock::wallclock);
    EXPECT_FALSE(fatal);
}

TEST(clock_option, reader_treats_reserved_clock_value_as_monotonic_without_failing)
{
    bool fatal = true;
    EXPECT_EQ(parsed_clock_for_flags(0x2 | (3U << 3), fatal), stream_clock::monotonic);
    EXPECT_FALSE(fatal);
}

TEST(clock_option, with_stream_clock_keeps_unwind_bits)
{
    using cpu_snap::core::with_stream_clock;
    EXPECT_EQ(with_stream_clock(0x4, stream_clock::boottime), 0x4U | 0x8U);
    EXPECT_EQ(with_stream_clock(0x2 | 0x18U, stream_clock::monotonic), 0x2U);
}

}  // namespace
