// Unit tests for cpu_snap::sampling::triggers::perf::kernel_drain epoll wiring.
//
// drain_all / sample decoding are covered end-to-end by the
// sampler_integration suite; these tests focus on the new
// wait() entry point and the epoll-fd lifecycle.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <ctime>
#include <bit>
#include <cstdint>
#include <cstring>
#include <future>
#include <thread>
#include <span>
#include <vector>

#include <linux/perf_event.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cpu_snap/sampling/unwind/kernel_callchain_unwinder.hpp"
#include "cpu_snap/sampling/unwind/perf_regs.hpp"
#include "cpu_snap/sampling/unwind_mode.hpp"
#include "cpu_snap/sampling/triggers/perf/kernel_drain.hpp"
#include "cpu_snap/sampling/triggers/perf/open_error.hpp"
#include "cpu_snap/sampling/triggers/perf/ring_sizing.hpp"

namespace
{

using cpu_snap::sampling::triggers::perf::kernel_drain;
using cpu_snap::sampling::triggers::perf::perf_sample_decoder;

inline constexpr std::uint32_t k_default_sample_freq_hz       = 500;
inline constexpr std::uint32_t k_high_sample_freq_hz          = 4000;
inline constexpr std::uint32_t k_default_callchain_depth      = 16;
inline constexpr std::uint32_t k_default_kernel_pages_per_tid = 4;
inline constexpr int           k_wait_short_ms                = 50;
inline constexpr int           k_wait_idle_ms                 = 200;
inline constexpr int           k_wait_slack_ms                = 60;
inline constexpr int           k_burn_ms                      = 100;
inline constexpr int           k_burn_inner_iters             = 1024;

perf_sample_decoder::decode_scratch&
scratch16()
{
    static perf_sample_decoder::decode_scratch scratch{ k_default_callchain_depth };
    return scratch;
}

kernel_drain::config
default_drain_config() noexcept
{
    kernel_drain::config cfg{};
    cfg.sample_freq_hz       = k_default_sample_freq_hz;
    cfg.callchain_depth      = k_default_callchain_depth;
    cfg.kernel_pages_per_tid = k_default_kernel_pages_per_tid;
    cfg.exclude_kernel       = true;
    cfg.exclude_hv           = true;
    cfg.dwarf_stack_bytes    = 0;
    cfg.unwind               = cpu_snap::sampling::unwind_mode::frame_pointer;
    return cfg;
}

bool
perf_open_available() noexcept
{
    struct perf_event_attr attr
    {};
    attr.size           = sizeof(attr);
    attr.type           = PERF_TYPE_SOFTWARE;
    attr.config         = PERF_COUNT_SW_CPU_CLOCK;
    attr.disabled       = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv     = 1;
    const int fd        = static_cast<int>(::syscall(SYS_perf_event_open, &attr,
                                                     /*pid=*/0, /*cpu=*/-1,
                                                     /*group_fd=*/-1, PERF_FLAG_FD_CLOEXEC));
    if(fd < 0)
    {
        return false;
    }
    ::close(fd);
    return true;
}

pid_t
self_tid() noexcept
{
    return static_cast<pid_t>(::syscall(SYS_gettid));
}

// Spin briefly on the calling thread so PERF_COUNT_SW_CPU_CLOCK has at
// least one chance to fire before the test queries epoll.
void
burn_cpu_for(std::chrono::milliseconds duration) noexcept
{
    volatile double acc   = 0.0;
    const auto      until = std::chrono::steady_clock::now() + duration;
    while(std::chrono::steady_clock::now() < until)
    {
        for(int i = 0; i < k_burn_inner_iters; ++i)
        {
            acc += static_cast<double>(i);
        }
    }
}

}  // namespace

TEST(KernelDrain, WaitTimesOutWithNoRings)
{
    kernel_drain drain(default_drain_config());

    const auto start      = std::chrono::steady_clock::now();
    const int  returned   = drain.wait(k_wait_short_ms);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count();

    EXPECT_LE(returned, 0);
    EXPECT_LE(elapsed_ms, k_wait_short_ms + k_wait_slack_ms);
}

TEST(KernelDrain, OpenAndCloseTidRegistersWithEpoll)
{
    if(!perf_open_available())
    {
        GTEST_SKIP() << "perf_event_open not permitted";
    }

    kernel_drain drain(default_drain_config());

    const pid_t tid = self_tid();
    EXPECT_NO_THROW(drain.open_tid(tid));
    EXPECT_TRUE(drain.is_tracked(tid));
    EXPECT_EQ(drain.tracked_tids(), 1U);

    drain.close_tid(tid);
    EXPECT_FALSE(drain.is_tracked(tid));
    EXPECT_EQ(drain.tracked_tids(), 0U);
}

TEST(KernelDrain, WaitReturnsReadyWhenSamplesQueue)
{
    if(!perf_open_available())
    {
        GTEST_SKIP() << "perf_event_open not permitted";
    }

    auto cfg           = default_drain_config();
    cfg.sample_freq_hz = k_high_sample_freq_hz;
    kernel_drain drain(cfg);
    drain.open_tid(self_tid());

    burn_cpu_for(std::chrono::milliseconds{ k_burn_ms });

    const int ready = drain.wait(k_wait_short_ms);
    EXPECT_GT(ready, 0);

    drain.close_all();
}

TEST(KernelDrain, CloseAllReleasesEpollFdAndIsIdempotent)
{
    kernel_drain drain(default_drain_config());
    EXPECT_NO_THROW(drain.close_all());
    EXPECT_EQ(drain.tracked_tids(), 0U);

    // Second close_all is a no-op; wait() on a closed drain returns 0.
    EXPECT_NO_THROW(drain.close_all());
    EXPECT_EQ(drain.wait(0), 0);
}

namespace
{

using cpu_snap::sampling::unwind::capture_kind;
using cpu_snap::sampling::unwind::machine_snapshot;
using cpu_snap::sampling::unwind::k_perf_dwarf_n_regs_x86_64;

struct recording_unwinder
{
    std::size_t unwind(const machine_snapshot& snap, std::span<std::uint64_t>) noexcept
    {
        called       = true;
        stack_bytes  = snap.stack.size();
        ready_frames = snap.ready_frames.size();
        return 0;
    }

    bool        called       = false;
    std::size_t stack_bytes  = 0;
    std::size_t ready_frames = 0;
};

struct dwarf_recording_unwinder : recording_unwinder
{
    static constexpr capture_kind required_capture() noexcept
    {
        return capture_kind::regs_and_stack;
    }
};

struct callchain_recording_unwinder : recording_unwinder
{
    static constexpr capture_kind required_capture() noexcept
    {
        return capture_kind::kernel_callchain;
    }
};

// Backing buffer is larger than the record so an over-read stays inside
// allocated memory instead of crashing.
class record_builder
{
public:
    record_builder() : m_bytes(sizeof(perf_event_header) + 4096 * 4, 0) {}

    record_builder& u64(std::uint64_t v)
    {
        std::memcpy(m_bytes.data() + m_used, &v, sizeof(v));
        m_used += sizeof(v);
        return *this;
    }

    record_builder& u64_at(std::size_t offset, std::uint64_t v)
    {
        std::memcpy(m_bytes.data() + offset, &v, sizeof(v));
        return *this;
    }

    record_builder& skip(std::size_t n)
    {
        m_used += n;
        return *this;
    }

    const perf_event_header* build()
    {
        perf_event_header hdr{};
        hdr.type = PERF_RECORD_SAMPLE;
        hdr.size = static_cast<std::uint16_t>(m_used);
        std::memcpy(m_bytes.data(), &hdr, sizeof(hdr));
        return reinterpret_cast<const perf_event_header*>(m_bytes.data());
    }

    record_builder& sample_prefix()
    {
        m_used = sizeof(perf_event_header);
        return u64(0x1000).u64(0).u64(0);
    }

private:
    std::vector<std::uint8_t> m_bytes;
    std::size_t               m_used = 0;
};

struct sink_log
{
    mutable std::size_t samples = 0;
    void operator()(pid_t, std::uint64_t,
                    std::span<const std::uint64_t>) const noexcept
    {
        ++samples;
    }
};

constexpr std::uint32_t k_dwarf_stack = 64;

record_builder
dwarf_record(std::uint64_t dumped)
{
    record_builder rb;
    rb.sample_prefix().u64(2 /*abi*/).skip(8 * k_perf_dwarf_n_regs_x86_64).u64(dumped);
    return rb;
}

}  // namespace

TEST(KernelDrainDecode, DwarfRecordWithValidStackReachesUnwinder)
{
    auto rb = dwarf_record(k_dwarf_stack);
    rb.skip(k_dwarf_stack).u64(k_dwarf_stack);

    dwarf_recording_unwinder unw{};
    sink_log                 log;
    perf_sample_decoder::process_sample(rb.build(), scratch16(), unw, log);

    EXPECT_TRUE(unw.called);
    EXPECT_EQ(unw.stack_bytes, k_dwarf_stack);
    EXPECT_EQ(log.samples, 1U);
}

TEST(KernelDrainDecode, DwarfStackSizeBeyondRecordIsNotTrusted)
{
    auto rb = dwarf_record(4096);
    constexpr std::size_t dyn_offset = sizeof(perf_event_header) + 24 + 8 +
                                       8 * k_perf_dwarf_n_regs_x86_64 + 8 + 4096;
    rb.u64_at(dyn_offset, 4096);

    dwarf_recording_unwinder unw{};
    sink_log                 log;
    perf_sample_decoder::process_sample(rb.build(), scratch16(), unw, log);

    EXPECT_FALSE(unw.called);
    EXPECT_EQ(log.samples, 1U);
}

TEST(KernelDrainDecode, DwarfDynSizeBeyondDumpedSizeIsClamped)
{
    auto rb = dwarf_record(k_dwarf_stack);
    rb.skip(k_dwarf_stack).u64(4096);

    dwarf_recording_unwinder unw{};
    sink_log                 log;
    perf_sample_decoder::process_sample(rb.build(), scratch16(), unw, log);

    EXPECT_TRUE(unw.called);
    EXPECT_EQ(unw.stack_bytes, k_dwarf_stack);
}

TEST(KernelDrainDecode, DwarfRecordTruncatedInsideRegsIsDropped)
{
    record_builder rb;
    rb.sample_prefix().u64(2).skip(8);

    dwarf_recording_unwinder unw{};
    sink_log                 log;
    perf_sample_decoder::process_sample(rb.build(), scratch16(), unw, log);

    EXPECT_FALSE(unw.called);
}

TEST(KernelDrainDecode, RecordShorterThanSamplePrefixIsDropped)
{
    record_builder rb;
    rb.sample_prefix();
    auto* hdr = const_cast<perf_event_header*>(rb.build());
    hdr->size = sizeof(perf_event_header) + 8;

    callchain_recording_unwinder unw{};
    sink_log                     log;
    perf_sample_decoder::process_sample(hdr, scratch16(), unw, log);

    EXPECT_EQ(log.samples, 0U);
}

TEST(KernelDrainDecode, CallchainLengthBeyondRecordIsClamped)
{
    record_builder rb;
    rb.sample_prefix().u64(1000).u64(0x2000).u64(0x3000);

    callchain_recording_unwinder unw{};
    sink_log                     log;
    perf_sample_decoder::process_sample(rb.build(), scratch16(), unw, log);

    EXPECT_TRUE(unw.called);
    EXPECT_EQ(unw.ready_frames, 2U);
}

TEST(KernelDrainDecode, CallchainLongerThanFormerFixedCapIsDecodedInFull)
{
    constexpr std::uint64_t k_chain_len = 300;
    constexpr std::uint64_t k_first_ip  = 0x1000;

    record_builder rb;
    rb.sample_prefix().u64(k_chain_len);
    for(std::uint64_t i = 0; i < k_chain_len; ++i)
    {
        rb.u64(k_first_ip + i);
    }

    perf_sample_decoder::decode_scratch scratch{ 512 };
    callchain_recording_unwinder unw{};
    sink_log                     log;
    perf_sample_decoder::process_sample(rb.build(), scratch, unw, log);

    EXPECT_EQ(unw.ready_frames, k_chain_len);
}

TEST(KernelDrainDecode, CallchainIsCutAtScratchDepth)
{
    constexpr std::uint64_t k_chain_len = 40;

    record_builder rb;
    rb.sample_prefix().u64(k_chain_len);
    for(std::uint64_t i = 0; i < k_chain_len; ++i)
    {
        rb.u64(0x1000 + i);
    }

    callchain_recording_unwinder unw{};
    sink_log                     log;
    perf_sample_decoder::process_sample(rb.build(), scratch16(), unw, log);

    EXPECT_EQ(unw.ready_frames, 16U);
}

namespace
{

std::size_t
drain_sample_count(kernel_drain& drain)
{
    cpu_snap::sampling::unwind::kernel_callchain_unwinder unwinder;
    std::size_t                                              samples = 0;
    drain.drain_all(
        unwinder,
        [&samples](pid_t, std::uint64_t,
                   std::span<const std::uint64_t>) noexcept { ++samples; },
        [](std::uint64_t) noexcept {});
    return samples;
}

}  // namespace

TEST(KernelDrain, WaitDoesNotSpinOnRingOfExitedThread)
{
    if(!perf_open_available())
    {
        GTEST_SKIP() << "perf_event_open not permitted";
    }

    kernel_drain drain(default_drain_config());

    std::promise<pid_t> tid_ready;
    std::promise<void>  may_exit;
    std::thread         worker([&] {
        tid_ready.set_value(self_tid());
        may_exit.get_future().wait();
    });
    drain.open_tid(tid_ready.get_future().get());
    may_exit.set_value();
    worker.join();

    (void) drain.wait(0);

    const auto start      = std::chrono::steady_clock::now();
    (void) drain.wait(k_wait_idle_ms);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count();

    EXPECT_GE(elapsed_ms, k_wait_idle_ms / 2);
}

TEST(KernelDrain, TidOpenedWhilePausedStaysDisabledUntilResume)
{
    if(!perf_open_available())
    {
        GTEST_SKIP() << "perf_event_open not permitted";
    }

    auto cfg           = default_drain_config();
    cfg.sample_freq_hz = k_high_sample_freq_hz;
    kernel_drain drain(cfg);

    drain.disable_all();
    drain.open_tid(self_tid());
    burn_cpu_for(std::chrono::milliseconds{ k_burn_ms });
    EXPECT_EQ(drain_sample_count(drain), 0U);

    drain.enable_all();
    burn_cpu_for(std::chrono::milliseconds{ k_burn_ms });
    EXPECT_GT(drain_sample_count(drain), 0U);
}

TEST(KernelDrain, OpenTidWorksAfterCloseAll)
{
    if(!perf_open_available())
    {
        GTEST_SKIP() << "perf_event_open not permitted";
    }

    auto cfg           = default_drain_config();
    cfg.sample_freq_hz = k_high_sample_freq_hz;
    kernel_drain drain(cfg);

    drain.open_tid(self_tid());
    drain.close_all();

    EXPECT_NO_THROW(drain.open_tid(self_tid()));
    EXPECT_EQ(drain.tracked_tids(), 1U);

    burn_cpu_for(std::chrono::milliseconds{ k_burn_ms });
    EXPECT_GT(drain.wait(k_wait_short_ms), 0);
    EXPECT_GT(drain_sample_count(drain), 0U);
}

TEST(KernelDrain, ChildCloseAllKeepsParentEpollRegistrations)
{
    if(!perf_open_available())
    {
        GTEST_SKIP() << "perf_event_open not permitted";
    }

    auto cfg           = default_drain_config();
    cfg.sample_freq_hz = k_high_sample_freq_hz;
    kernel_drain drain(cfg);
    drain.open_tid(self_tid());

    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if(child == 0)
    {
        // The epoll instance is shared across fork; the child must not unregister the parent's rings.
        drain.close_all();
        ::_exit(0);
    }

    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));

    burn_cpu_for(std::chrono::milliseconds{ k_burn_ms });
    EXPECT_GT(drain.wait(k_wait_idle_ms), 0);
}

TEST(PerfOpenError, PermissionDeniedMentionsParanoidValueAndSignalBackend)
{
    const auto text = cpu_snap::sampling::triggers::perf::describe_perf_open_failure(
        1234, EACCES, 4);

    EXPECT_THAT(text, ::testing::HasSubstr("tid=1234"));
    EXPECT_THAT(text, ::testing::HasSubstr("perf_event_paranoid"));
    EXPECT_THAT(text, ::testing::HasSubstr("currently 4"));
    EXPECT_THAT(text, ::testing::HasSubstr("backend_kind::signal"));
}

TEST(PerfOpenError, PermissionDeniedWithUnknownParanoidOmitsValue)
{
    const auto text = cpu_snap::sampling::triggers::perf::describe_perf_open_failure(
        7, EPERM, std::nullopt);

    EXPECT_THAT(text, ::testing::HasSubstr("perf_event_paranoid"));
    EXPECT_THAT(text, ::testing::Not(::testing::HasSubstr("currently")));
}

TEST(PerfOpenError, OtherErrorsKeepOnlyTheCallDescription)
{
    EXPECT_EQ(cpu_snap::sampling::triggers::perf::describe_perf_open_failure(7, ENOENT, 4),
              "perf_event_open(tid=7)");
}

namespace sizing = cpu_snap::sampling::triggers::perf;
using cpu_snap::sampling::unwind_mode;

TEST(RingSizing, DefaultPagesHoldEightRecordsOfDefaultSnapshot)
{
    constexpr std::size_t page = 4096;
    const auto            pages = sizing::default_ring_pages(unwind_mode::dwarf, 32768, page);

    EXPECT_EQ(std::popcount(pages), 1);
    EXPECT_GE(static_cast<std::size_t>(pages) * page,
              8 * sizing::max_sample_record_bytes(unwind_mode::dwarf, 32768));
}

TEST(RingSizing, FramePointerDefaultKeepsEightPages)
{
    EXPECT_EQ(sizing::default_ring_pages(unwind_mode::frame_pointer, 0, 4096), 8U);
}

TEST(RingSizing, RingHoldsRecordOnlyWhenAtLeastOneRecordFits)
{
    EXPECT_FALSE(sizing::ring_holds_a_record(unwind_mode::dwarf, 32768, 8, 4096));
    EXPECT_TRUE(sizing::ring_holds_a_record(unwind_mode::dwarf, 32768, 16, 4096));
}

TEST(RingSizing, WakeupEventsScaleWithRingCapacityAndStayInRange)
{
    EXPECT_EQ(sizing::wakeup_events_for(unwind_mode::dwarf, 32768, 16, 4096), 1U);
    const auto big = sizing::wakeup_events_for(unwind_mode::frame_pointer, 0, 4096, 4096);
    EXPECT_EQ(big, sizing::k_max_wakeup_events);
    EXPECT_GE(sizing::wakeup_events_for(unwind_mode::dwarf, 32768, 1, 4096), 1U);
}

namespace
{

std::uint64_t
monotonic_ns() noexcept
{
    timespec ts{};
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

}  // namespace

TEST(KernelDrain, PerfAttrStampsSamplesWithMonotonicClock)
{
    const auto attr = kernel_drain::make_perf_attr(default_drain_config());

    EXPECT_EQ(attr.use_clockid, 1U);
    EXPECT_EQ(attr.clockid, CLOCK_MONOTONIC);
}

TEST(KernelDrain, SampleTimestampsLieInsideTheMonotonicCaptureWindow)
{
    if(!perf_open_available())
    {
        GTEST_SKIP() << "perf_event_open not permitted";
    }

    auto cfg           = default_drain_config();
    cfg.sample_freq_hz = k_high_sample_freq_hz;
    kernel_drain drain(cfg);

    const std::uint64_t before = monotonic_ns();
    drain.open_tid(self_tid());
    burn_cpu_for(std::chrono::milliseconds{ k_burn_ms });
    drain.disable_all();

    cpu_snap::sampling::unwind::kernel_callchain_unwinder unwinder;
    std::vector<std::uint64_t>                               stamps;
    drain.drain_all(
        unwinder,
        [&stamps](pid_t, std::uint64_t ts,
                  std::span<const std::uint64_t>) noexcept { stamps.push_back(ts); },
        [](std::uint64_t) noexcept {});
    const std::uint64_t after = monotonic_ns();

    ASSERT_FALSE(stamps.empty());
    for(const auto ts : stamps)
    {
        EXPECT_GE(ts, before);
        EXPECT_LE(ts, after);
    }
}
