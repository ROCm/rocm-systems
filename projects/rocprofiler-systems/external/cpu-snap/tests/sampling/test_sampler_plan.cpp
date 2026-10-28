#include <gtest/gtest.h>

#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <functional>

#include <unistd.h>

#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/sampling/sampler_plan.hpp"
#include "cpu_snap/sampling/triggers/perf/ring_sizing.hpp"

namespace
{

using cpu_snap::config_error;
using cpu_snap::environment_error;
using cpu_snap::sampling::backend_kind;
using cpu_snap::sampling::clock_kind;
using cpu_snap::sampling::perf_probe_result;
using cpu_snap::sampling::sampler;
using cpu_snap::sampling::unwind_mode;
using cpu_snap::sampling::internal::resolve_plan;
namespace perf = cpu_snap::sampling::triggers::perf;

int g_probe_calls = 0;

perf_probe_result
probe_ok(bool)
{
    ++g_probe_calls;
    return { true, 0 };
}

perf_probe_result
probe_denied(bool)
{
    ++g_probe_calls;
    return { false, EACCES };
}

class SamplerPlanTest : public ::testing::Test
{
protected:
    void SetUp() override { g_probe_calls = 0; }

    sampler::config make_config()
    {
        sampler::config cfg;
        cfg.streaming.sink = std::ref(m_sink);
        return cfg;
    }

    static std::size_t page_bytes()
    {
        return static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    }

    cpu_snap::core::memory_sink m_sink{ cpu_snap::core::memory_sink::config{
        .max_bytes = 1U << 20 } };
};

TEST_F(SamplerPlanTest, AutomaticWithWorkingProbePicksPerf)
{
    const auto plan = resolve_plan(make_config(), &probe_ok, false);
    EXPECT_EQ(plan.choice.kind, backend_kind::perf);
    EXPECT_TRUE(plan.choice.reason.empty());
    EXPECT_EQ(plan.cfg.backend, backend_kind::perf);
    EXPECT_EQ(g_probe_calls, 1);
}

TEST_F(SamplerPlanTest, AutomaticWhenDeniedFallsBackToSignalWithReason)
{
    auto cfg                = make_config();
    cfg.perf.sample_freq_hz = 777;
    const auto plan         = resolve_plan(std::move(cfg), &probe_denied, false);
    EXPECT_EQ(plan.choice.kind, backend_kind::signal);
    EXPECT_FALSE(plan.choice.reason.empty());
    EXPECT_EQ(plan.cfg.signal.sample_freq_hz, 777U);
}

TEST_F(SamplerPlanTest, AutomaticFallbackKeepsExplicitSignalFrequency)
{
    auto cfg                  = make_config();
    cfg.perf.sample_freq_hz   = 777;
    cfg.signal.sample_freq_hz = 321;
    const auto plan           = resolve_plan(std::move(cfg), &probe_denied, false);
    EXPECT_EQ(plan.cfg.signal.sample_freq_hz, 321U);
}

TEST_F(SamplerPlanTest, ExplicitPerfDeniedThrowsEnvironmentError)
{
    auto cfg    = make_config();
    cfg.backend = backend_kind::perf;
    EXPECT_THROW((void) resolve_plan(std::move(cfg), &probe_denied, false), environment_error);
}

TEST_F(SamplerPlanTest, ExplicitSignalNeverProbes)
{
    auto cfg    = make_config();
    cfg.backend = backend_kind::signal;
    const auto plan = resolve_plan(std::move(cfg), &probe_denied, false);
    EXPECT_EQ(plan.choice.kind, backend_kind::signal);
    EXPECT_TRUE(plan.choice.reason.empty());
    EXPECT_EQ(g_probe_calls, 0);
}

TEST_F(SamplerPlanTest, InjectedTriggerWithAutomaticIsPerfWithoutProbing)
{
    const auto plan = resolve_plan(make_config(), &probe_denied, true);
    EXPECT_EQ(plan.choice.kind, backend_kind::perf);
    EXPECT_TRUE(plan.choice.reason.empty());
    EXPECT_EQ(g_probe_calls, 0);
}

TEST_F(SamplerPlanTest, AutoBestResolvesToDwarfOnBothBackends)
{
    for(const auto kind : { backend_kind::perf, backend_kind::signal })
    {
        auto cfg    = make_config();
        cfg.backend = kind;
        ASSERT_EQ(cfg.perf.unwind, unwind_mode::auto_best);
        const auto plan = resolve_plan(std::move(cfg), &probe_ok, false);
        EXPECT_EQ(plan.cfg.perf.unwind, unwind_mode::dwarf);
        EXPECT_TRUE(plan.unwind_was_auto);
    }
}

TEST_F(SamplerPlanTest, ExplicitUnwindModeIsKept)
{
    for(const auto mode : { unwind_mode::frame_pointer, unwind_mode::dwarf })
    {
        auto cfg         = make_config();
        cfg.perf.unwind  = mode;
        const auto plan  = resolve_plan(std::move(cfg), &probe_ok, false);
        EXPECT_EQ(plan.cfg.perf.unwind, mode);
        EXPECT_FALSE(plan.unwind_was_auto);
    }
}

TEST_F(SamplerPlanTest, ZeroCallchainDepthThrows)
{
    auto cfg                 = make_config();
    cfg.perf.callchain_depth = 0;
    EXPECT_THROW((void) resolve_plan(std::move(cfg), &probe_ok, false), config_error);
}

TEST_F(SamplerPlanTest, OversizedCallchainDepthThrows)
{
    auto cfg                 = make_config();
    cfg.perf.callchain_depth = cpu_snap::core::k_max_callchain_depth + 1;
    EXPECT_THROW((void) resolve_plan(std::move(cfg), &probe_ok, false), config_error);
}

TEST_F(SamplerPlanTest, NonPowerOfTwoKernelPagesThrows)
{
    auto cfg                      = make_config();
    cfg.perf.kernel_pages_per_tid = 12;
    EXPECT_THROW((void) resolve_plan(std::move(cfg), &probe_ok, false), config_error);
}

TEST_F(SamplerPlanTest, MisalignedDwarfStackBytesThrows)
{
    auto cfg                  = make_config();
    cfg.perf.unwind           = unwind_mode::dwarf;
    cfg.perf.dwarf_stack_bytes = 1001;
    EXPECT_THROW((void) resolve_plan(std::move(cfg), &probe_ok, false), config_error);
}

TEST_F(SamplerPlanTest, ZeroDwarfStackBytesThrowsOnPerf)
{
    auto cfg                   = make_config();
    cfg.backend                = backend_kind::perf;
    cfg.perf.unwind            = unwind_mode::dwarf;
    cfg.perf.dwarf_stack_bytes = 0;
    EXPECT_THROW((void) resolve_plan(std::move(cfg), &probe_ok, false), config_error);
}

TEST_F(SamplerPlanTest, MissingStreamingSinkThrows)
{
    sampler::config cfg;
    EXPECT_THROW((void) resolve_plan(std::move(cfg), &probe_ok, false), config_error);
}

TEST_F(SamplerPlanTest, SignalOffsetOutsideRealtimeRangeThrows)
{
    auto cfg                     = make_config();
    cfg.backend                  = backend_kind::signal;
    cfg.signal.signal_rt_offset  = static_cast<std::uint32_t>(SIGRTMAX - SIGRTMIN) + 1;
    EXPECT_THROW((void) resolve_plan(std::move(cfg), &probe_ok, false), config_error);
}

TEST_F(SamplerPlanTest, PerfDwarfRingIsSizedToHoldARecord)
{
    auto cfg                   = make_config();
    cfg.backend                = backend_kind::perf;
    cfg.perf.unwind            = unwind_mode::dwarf;
    cfg.perf.dwarf_stack_bytes = 32768;
    const auto plan            = resolve_plan(std::move(cfg), &probe_ok, false);
    const auto pages           = plan.cfg.perf.kernel_pages_per_tid;
    EXPECT_GT(pages, 0U);
    EXPECT_TRUE(perf::ring_holds_a_record(unwind_mode::dwarf, 32768, pages, page_bytes()));
}

TEST_F(SamplerPlanTest, FramePointerKeepsExplicitPages)
{
    auto cfg                      = make_config();
    cfg.backend                   = backend_kind::perf;
    cfg.perf.unwind               = unwind_mode::frame_pointer;
    cfg.perf.kernel_pages_per_tid = 16;
    const auto plan               = resolve_plan(std::move(cfg), &probe_ok, false);
    EXPECT_EQ(plan.cfg.perf.kernel_pages_per_tid, 16U);
}

TEST_F(SamplerPlanTest, TooSmallExplicitPagesWithDwarfThrows)
{
    auto cfg                      = make_config();
    cfg.backend                   = backend_kind::perf;
    cfg.perf.unwind               = unwind_mode::dwarf;
    cfg.perf.dwarf_stack_bytes    = 32768;
    cfg.perf.kernel_pages_per_tid = 1;
    EXPECT_THROW((void) resolve_plan(std::move(cfg), &probe_ok, false), config_error);
}

TEST(SamplerPlanHelpersTest, StreamClockMapping)
{
    using cpu_snap::core::stream_clock;
    using cpu_snap::sampling::internal::stream_clock_for;
    EXPECT_EQ(stream_clock_for(clock_kind::monotonic), stream_clock::monotonic);
    EXPECT_EQ(stream_clock_for(clock_kind::boottime), stream_clock::boottime);
    EXPECT_EQ(stream_clock_for(clock_kind::wallclock), stream_clock::wallclock);
}

TEST(SamplerPlanHelpersTest, StreamingFlagsCarryUnwindAndClockBits)
{
    using cpu_snap::core::stream_clock;
    using cpu_snap::core::with_stream_clock;
    using cpu_snap::sampling::internal::streaming_flags_for;
    EXPECT_EQ(streaming_flags_for(unwind_mode::frame_pointer, clock_kind::monotonic),
              with_stream_clock(0x2U, stream_clock::monotonic));
    EXPECT_EQ(streaming_flags_for(unwind_mode::dwarf, clock_kind::boottime),
              with_stream_clock(0x4U, stream_clock::boottime));
    EXPECT_EQ(streaming_flags_for(unwind_mode::dwarf, clock_kind::wallclock),
              with_stream_clock(0x4U, stream_clock::wallclock));
}

TEST(SamplerPlanHelpersTest, CaptureKindTable)
{
    using cpu_snap::sampling::internal::capture_kind_for;
    using kind = cpu_snap::sampling::unwind::capture_kind;
    EXPECT_EQ(capture_kind_for(unwind_mode::dwarf, backend_kind::perf), kind::regs_and_stack);
    EXPECT_EQ(capture_kind_for(unwind_mode::dwarf, backend_kind::signal), kind::regs_and_stack);
    EXPECT_EQ(capture_kind_for(unwind_mode::frame_pointer, backend_kind::perf),
              kind::kernel_callchain);
    EXPECT_EQ(capture_kind_for(unwind_mode::frame_pointer, backend_kind::signal),
              kind::regs_and_stack);
}

}  // namespace
