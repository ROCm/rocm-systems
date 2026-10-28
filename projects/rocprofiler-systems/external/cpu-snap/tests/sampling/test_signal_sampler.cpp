// Deterministic tests for the signal/timer backend; needs no
// perf_event_paranoid privilege, so every case runs unconditionally.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <functional>
#include <ios>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <csignal>
#include <pthread.h>
#include <malloc.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cpu_snap/error.hpp"
#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/sampling/triggers/signal/backend.hpp"
#include "cpu_snap/sampling/triggers/signal/ring_table.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"

namespace
{

// A 2-deep known call chain. busy_inner is the hot leaf: the timer signal will
// most often interrupt this tight loop, so its address range is what the leaf
// IP should fall into.
[[gnu::noinline]] double
busy_inner(double seed, std::uint64_t iters) noexcept
{
    double acc = seed;
    for(std::uint64_t i = 0; i < iters; ++i)
    {
        acc = acc * 1.0000001 + static_cast<double>(i & 0xFFFF);
    }
    return acc;
}

[[gnu::noinline]] double
busy_outer(double seed, std::uint64_t iters) noexcept
{
    return busy_inner(seed + 1.0, iters);
}

void
spin_worker(std::atomic<bool>& stop) noexcept
{
    double acc = 0.0;
    while(!stop.load(std::memory_order_relaxed))
    {
        acc += busy_outer(acc, 200'000);
    }
    // Defeat dead-code elimination of acc without doing real I/O on the path.
    if(acc == 12345.6789)
    {
        std::printf("unreachable %f\n", acc);
    }
}

// Dedicated 3-deep chain for the frame-pointer integration test. A genuine
// leaf function's frame-pointer prologue is elided by GCC at -O2 even under
// -fno-omit-frame-pointer (verified empirically: busy_inner above has no
// rbp setup at all), which would make fp_busy_inner's caller invisible to a
// naive rbp walk. fp_frame_barrier's call breaks that leaf-ness so every
// level of this chain gets a real, walkable frame.
[[gnu::noinline]] void
fp_frame_barrier() noexcept
{
    asm volatile("" ::: "memory");
}

[[gnu::noinline]] double
fp_busy_inner(double seed, std::uint64_t iters) noexcept
{
    double acc = seed;
    for(std::uint64_t i = 0; i < iters; ++i)
    {
        acc = acc * 1.0000001 + static_cast<double>(i & 0xFFFF);
    }
    fp_frame_barrier();
    return acc;
}

[[gnu::noinline]] double
fp_busy_outer(double seed, std::uint64_t iters) noexcept
{
    return fp_busy_inner(seed + 1.0, iters);
}

void
fp_spin_worker(std::atomic<bool>& stop) noexcept
{
    double acc = 0.0;
    while(!stop.load(std::memory_order_relaxed))
    {
        acc += fp_busy_outer(acc, 200'000);
    }
    if(acc == 12345.6789)
    {
        std::printf("unreachable %f\n", acc);
    }
}

// Address of a noinline function as an integer. Taking the address forces the
// symbol to be emitted; the leaf IP at sample time should sit at-or-after this.
std::uint64_t
func_addr(double (*fn)(double, std::uint64_t)) noexcept
{
    return reinterpret_cast<std::uint64_t>(fn);
}

// Byte bound for the per-test memory_sink. These tests run for at most a
// few hundred milliseconds each, so this is ample headroom.
constexpr std::size_t k_test_sink_max_bytes = 64U * 1024U * 1024U;

cpu_snap::sampling::sampler::config
make_signal_cfg(std::uint32_t freq_hz, cpu_snap::core::sink_fn sink)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.backend                    = cpu_snap::sampling::backend_kind::signal;
    cfg.signal.sample_freq_hz      = freq_hz;
    cfg.signal.stack_snapshot_bytes = 8192;
    cfg.signal.ring_capacity       = 1024;
    cfg.threads.rescan_period      = std::chrono::milliseconds{ 25 };
    cfg.streaming.sink             = std::move(sink);
    return cfg;
}

cpu_snap::sampling::sampler::config
make_perf_cfg(std::uint32_t freq_hz, cpu_snap::core::sink_fn sink)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.backend             = cpu_snap::sampling::backend_kind::perf;
    cfg.perf.sample_freq_hz = freq_hz;
    cfg.threads.rescan_period = std::chrono::milliseconds{ 25 };
    cfg.streaming.sink        = std::move(sink);
    return cfg;
}

// RT offset used by the sigqueue-exhaustion regression. The low RT signals
// (SIGRTMIN..SIGRTMIN+2) are reserved by glibc/NPTL and the kernel reclaims
// their timer-queued instances on timer_delete; an application-range offset
// reproduces the leak deterministically.
constexpr std::uint32_t k_exhaustion_rt_offset = 4;
const int k_exhaustion_signal_no = SIGRTMIN + static_cast<int>(k_exhaustion_rt_offset);

// Run one short signal-backend lifecycle over @p worker_stop's busy thread.
// Returns the total samples captured. Throws on backend setup failure (the
// lifecycle bug surfaced here as a timer_create EAGAIN throw).
std::uint64_t
run_signal_lifecycle(std::uint32_t freq_hz, std::chrono::milliseconds dwell)
{
    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
    cpu_snap::sampling::sampler smp{ make_signal_cfg(freq_hz, std::ref(sink)) };
    smp.start();
    std::this_thread::sleep_for(dwell);
    smp.stop();
    return smp.total_samples();
}

// Config that attaches sibling threads aggressively (fast rescan, main thread
// excluded so the sampled set is the sibling pool) for the sigqueue-exhaustion
// regression: it must reliably arm a per-thread timer against each blocker so an
// expiration can queue into the blocker's (blocked) pending set.
cpu_snap::sampling::sampler::config
make_aggressive_signal_cfg(std::uint32_t freq_hz, cpu_snap::core::sink_fn sink)
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.backend                     = cpu_snap::sampling::backend_kind::signal;
    cfg.signal.sample_freq_hz       = freq_hz;
    cfg.signal.signal_rt_offset     = k_exhaustion_rt_offset;
    cfg.signal.stack_snapshot_bytes = 4096;
    cfg.signal.ring_capacity        = 8192;
    cfg.threads.watch_new_threads   = true;
    cfg.threads.include_main_thread = false;
    cfg.threads.rescan_period       = std::chrono::milliseconds{ 3 };
    cfg.streaming.sink              = std::move(sink);
    return cfg;
}

}  // namespace

// Construction / config validation. No privileges, no timers armed.

// Cheap no-op sink for config-validation tests that never call start().
cpu_snap::core::sink_fn
noop_sink()
{
    return [](std::span<const std::byte>) {};
}

TEST(SignalSamplerConfigTest, CtorAcceptsSignalBackend)
{
    EXPECT_NO_THROW(cpu_snap::sampling::sampler{ make_signal_cfg(500, noop_sink()) });
}

TEST(SignalSamplerConfigTest, CtorRejectsZeroFrequency)
{
    auto cfg = make_signal_cfg(0, noop_sink());
    EXPECT_THROW(cpu_snap::sampling::sampler{ std::move(cfg) },
                 cpu_snap::config_error);
}

TEST(SignalSamplerConfigTest, CtorRejectsUnalignedStackBytes)
{
    auto cfg                       = make_signal_cfg(500, noop_sink());
    cfg.signal.stack_snapshot_bytes = 8191;  // not a multiple of 8
    EXPECT_THROW(cpu_snap::sampling::sampler{ std::move(cfg) },
                 cpu_snap::config_error);
}

TEST(SignalSamplerConfigTest, CtorRejectsTinyRing)
{
    auto cfg                 = make_signal_cfg(500, noop_sink());
    cfg.signal.ring_capacity = 1;
    EXPECT_THROW(cpu_snap::sampling::sampler{ std::move(cfg) },
                 cpu_snap::config_error);
}

// Signal + frame_pointer used to throw unconditionally (the signal trigger
// has no kernel-walked-callchain capability, and capture_kind_for() did not
// used to know that frame_pointer means something different for signal than
// for perf). fp_copy_unwinder + the backend-aware capture_kind mapping make
// this a valid, constructible pairing.
TEST(SignalSamplerConfigTest, CtorAcceptsSignalPlusFramePointer)
{
    auto cfg           = make_signal_cfg(500, noop_sink());
    cfg.perf.unwind    = cpu_snap::sampling::unwind_mode::frame_pointer;
    EXPECT_NO_THROW(cpu_snap::sampling::sampler{ std::move(cfg) });
}

// auto_best must still resolve to dwarf for the signal backend, unchanged
// from before this PR: fp_copy_unwinder's naive rbp walk is not safe to pick
// automatically for an arbitrary target binary (most ordinary -O2/-O3 code
// omits frame pointers), so signal's default stays dwarf. frame_pointer
// remains explicitly selectable (see CtorAcceptsSignalPlusFramePointer
// above) - only the automatic default is unaffected by that new capability.
TEST(SignalSamplerConfigTest, AutoBestStillResolvesToDwarfForSignalBackend)
{
    cpu_snap::sampling::sampler smp{ make_signal_cfg(500, noop_sink()) };
    EXPECT_EQ(smp.resolved_unwind_mode(), cpu_snap::sampling::unwind_mode::dwarf);
}

// End-to-end capture. Runs unconditionally (no perf gate).

TEST(SignalSamplerTest, CollectsSamplesForBusyThread)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
    cpu_snap::sampling::sampler smp{ make_signal_cfg(1000, std::ref(sink)) };
    smp.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 400 });
    smp.stop();

    stop.store(true);
    worker.join();

    EXPECT_GT(smp.total_samples(), 0U);

    cpu_snap::core::sample_reconstructor reconstructor;
    reconstructor.feed(sink.bytes());
    std::size_t stored = 0;
    for(const auto& [tid, vec] : reconstructor.samples())
    {
        stored += vec.size();
    }
    EXPECT_GT(stored, 0U);
    EXPECT_LE(stored, smp.total_samples());
}

TEST(SignalSamplerTest, LeafIpFallsInsideKnownWorkload)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
    cpu_snap::sampling::sampler smp{ make_signal_cfg(2000, std::ref(sink)) };
    smp.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 500 });
    smp.stop();

    stop.store(true);
    worker.join();

    ASSERT_GT(smp.total_samples(), 0U);

    cpu_snap::core::sample_reconstructor reconstructor;
    reconstructor.feed(sink.bytes());

    // The hot leaf is busy_inner; allow a generous code window covering the
    // whole 2-function chain (functions are small and adjacent in the TU, but
    // we do not rely on ordering -- accept either known frame).
    const std::uint64_t inner = func_addr(&busy_inner);
    const std::uint64_t outer = func_addr(&busy_outer);
    constexpr std::uint64_t k_window = 0x2000;  // 8 KiB per function: ample

    bool hit = false;
    for(const auto& [tid, vec] : reconstructor.samples())
    {
        for(const auto& rs : vec)
        {
            const std::uint64_t ip = rs.ip;
            const bool in_inner = ip >= inner && ip < inner + k_window;
            const bool in_outer = ip >= outer && ip < outer + k_window;
            if(in_inner || in_outer)
            {
                hit = true;
                break;
            }
        }
        if(hit)
        {
            break;
        }
    }
    EXPECT_TRUE(hit) << "no sample leaf IP landed inside busy_inner/busy_outer "
                        "[" << std::hex << inner << ", " << outer << "]";
}

// PR3: drain_into now runs the DWARF unwinder off-handler and emits the full
// innermost-first chain. Assert that at least one sample recovers multiple
// frames and that the recovered IPs land inside the known busy_inner/busy_outer
// chain (same +/- window tolerance the dwarf_unwinder test uses).
//
// unwind_mode is pinned explicitly to dwarf (auto_best would resolve to the
// same thing for signal - see AutoBestStillResolvesToDwarfForSignalBackend -
// but pinning it keeps this test's intent explicit regardless of default
// changes elsewhere).
TEST(SignalSamplerTest, RecoversMultipleFramesViaDwarf)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
    auto cfg        = make_signal_cfg(2000, std::ref(sink));
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::dwarf;
    cpu_snap::sampling::sampler smp{ std::move(cfg) };
    ASSERT_EQ(smp.resolved_unwind_mode(), cpu_snap::sampling::unwind_mode::dwarf);
    smp.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 500 });
    smp.stop();

    stop.store(true);
    worker.join();

    ASSERT_GT(smp.total_samples(), 0U);

    cpu_snap::core::sample_reconstructor reconstructor;
    reconstructor.feed(sink.bytes());

    const std::uint64_t     inner    = func_addr(&busy_inner);
    const std::uint64_t     outer    = func_addr(&busy_outer);
    constexpr std::uint64_t k_window = 0x2000;  // 8 KiB per function

    auto in_chain = [&](std::uint64_t ipv) noexcept {
        const bool in_inner = ipv >= inner && ipv < inner + k_window;
        const bool in_outer = ipv >= outer && ipv < outer + k_window;
        return in_inner || in_outer;
    };

    std::size_t   max_frames_seen   = 0;
    bool          multi_in_chain    = false;
    for(const auto& [tid, vec] : reconstructor.samples())
    {
        for(const auto& rs : vec)
        {
            max_frames_seen = std::max(max_frames_seen, rs.frames.size());

            // Count how many of this sample's recovered frames sit inside the
            // known 2-deep chain. A correct unwind recovers both busy_inner and
            // busy_outer for at least some samples interrupted in the leaf.
            std::uint32_t in_chain_frames = 0;
            for(std::uint32_t i = 0; i < rs.frames.size(); ++i)
            {
                if(in_chain(rs.frames[i]))
                {
                    ++in_chain_frames;
                }
            }
            if(rs.frames.size() >= 2 && in_chain_frames >= 2)
            {
                multi_in_chain = true;
            }
        }
    }

    EXPECT_GE(max_frames_seen, 2U)
        << "no sample recovered >= 2 frames; DWARF unwind likely produced "
           "leaf-only fallbacks";
    EXPECT_TRUE(multi_in_chain)
        << "no sample recovered both busy_inner and busy_outer via DWARF "
           "[inner " << std::hex << inner << ", outer " << outer << "]";
}

[[gnu::noinline]] std::uint64_t
deep_recursion(std::uint32_t depth, std::atomic<bool>& stop) noexcept
{
    if(depth == 0)
    {
        std::uint64_t acc = 0;
        while(!stop.load(std::memory_order_relaxed))
        {
            acc += static_cast<std::uint64_t>(busy_inner(1.0, 20'000));
        }
        return acc;
    }
    volatile std::uint64_t pad[8] = { depth };
    const std::uint64_t    r      = deep_recursion(depth - 1, stop);
    return r + pad[0];
}

std::uint64_t
run_dwarf_sampler_live_reads(std::uint32_t snapshot_bytes, std::uint32_t depth,
                             std::size_t& total_samples)
{
    std::atomic<bool> stop{ false };
    std::thread       worker([&] { (void) deep_recursion(depth, stop); });

    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
    auto cfg                        = make_signal_cfg(2000, std::ref(sink));
    cfg.perf.unwind                 = cpu_snap::sampling::unwind_mode::dwarf;
    cfg.signal.stack_snapshot_bytes = snapshot_bytes;
    cfg.signal.ring_capacity        = 8;
    cpu_snap::sampling::sampler smp{ std::move(cfg) };
    smp.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 400 });
    smp.stop();
    stop.store(true);
    worker.join();

    total_samples = smp.total_samples();
    return smp.unwinder_live_stack_samples();
}

TEST(SignalSamplerTest, DeepStackWithSmallSnapshotCountsLiveStackReads)
{
    std::size_t         total = 0;
    const std::uint64_t live  = run_dwarf_sampler_live_reads(2048, 300, total);
    ASSERT_GT(total, 0U);
    EXPECT_GT(live, 0U);
    EXPECT_LE(live, total);
}

// The sampler also samples the idle test main thread, whose stack is about
// 1 MiB deep here, so "large" has to cover that whole stack.
constexpr std::uint32_t k_whole_stack_window = 4U * 1024U * 1024U;

TEST(SignalSamplerTest, ShallowStackWithLargeSnapshotCountsNoLiveStackReads)
{
    std::size_t         total = 0;
    const std::uint64_t live  = run_dwarf_sampler_live_reads(k_whole_stack_window, 4, total);
    ASSERT_GT(total, 0U);
    EXPECT_EQ(live, 0U);
}

// Real integration test for signal + frame_pointer (fp_copy_unwinder). Uses
// the dedicated fp_busy_inner -> fp_busy_outer -> fp_spin_worker chain (see
// fp_frame_barrier's comment above for why the shared busy_inner/busy_outer
// workload does not work here: GCC elides even -fno-omit-frame-pointer's
// prologue for a genuine leaf function at -O2). Symbolizes the recovered
// chain and asserts the actual resolved function sequence is innermost-first
// fp_busy_inner -> fp_busy_outer -> fp_spin_worker.
TEST(SignalSamplerTest, RecoversMultipleFramesViaFramePointer)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(fp_spin_worker, std::ref(stop));

    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
    auto cfg        = make_signal_cfg(2000, std::ref(sink));
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cpu_snap::sampling::sampler smp{ std::move(cfg) };
    ASSERT_EQ(smp.resolved_unwind_mode(), cpu_snap::sampling::unwind_mode::frame_pointer);
    smp.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 500 });
    smp.stop();

    stop.store(true);
    worker.join();

    ASSERT_GT(smp.total_samples(), 0U);

    cpu_snap::core::sample_reconstructor reconstructor;
    reconstructor.feed(sink.bytes());

    std::size_t                 max_frames_seen = 0;
    std::vector<std::uint64_t>  all_ips;
    std::vector<std::vector<std::uint64_t>> chains;
    for(const auto& [tid, vec] : reconstructor.samples())
    {
        for(const auto& rs : vec)
        {
            max_frames_seen = std::max(max_frames_seen, rs.frames.size());
            if(rs.frames.size() >= 3)
            {
                chains.emplace_back(rs.frames.begin(), rs.frames.end());
                all_ips.insert(all_ips.end(), chains.back().begin(), chains.back().end());
            }
        }
    }

    EXPECT_GE(max_frames_seen, 3U)
        << "no sample recovered >= 3 frames; fp_copy_unwinder likely produced "
           "leaf-only fallbacks";
    ASSERT_FALSE(chains.empty());

    cpu_snap::processing::symbols::symbolizer sym(
        cpu_snap::processing::symbols::symbolizer_config{ .include_binary_name = false });
    sym.preload(all_ips);

    bool found_ordered_chain = false;
    for(const auto& chain : chains)
    {
        int inner_idx  = -1;
        int outer_idx  = -1;
        int worker_idx = -1;
        for(std::size_t i = 0; i < chain.size(); ++i)
        {
            const std::string_view name = sym.resolve(chain[i]);
            if(inner_idx < 0 && name.find("fp_busy_inner") != std::string_view::npos)
                inner_idx = static_cast<int>(i);
            if(outer_idx < 0 && name.find("fp_busy_outer") != std::string_view::npos)
                outer_idx = static_cast<int>(i);
            if(worker_idx < 0 && name.find("fp_spin_worker") != std::string_view::npos)
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
           "fp_busy_inner -> fp_busy_outer -> fp_spin_worker";
}

// Confirms the DEFAULT experience for signal users is completely unchanged
// by this PR: a config that leaves unwind_mode at its auto_best default (no
// explicit setting at all) must still resolve to dwarf and still recover a
// real multi-frame call chain, exactly as before fp_copy_unwinder existed.
TEST(SignalSamplerTest, AutoBestDefaultStillRecoversCallChainViaDwarf)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
    // unwind_mode left at auto_best
    cpu_snap::sampling::sampler smp{ make_signal_cfg(2000, std::ref(sink)) };
    ASSERT_EQ(smp.resolved_unwind_mode(), cpu_snap::sampling::unwind_mode::dwarf);
    smp.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 500 });
    smp.stop();

    stop.store(true);
    worker.join();

    ASSERT_GT(smp.total_samples(), 0U);

    cpu_snap::core::sample_reconstructor reconstructor;
    reconstructor.feed(sink.bytes());

    std::size_t max_frames_seen = 0;
    for(const auto& [tid, vec] : reconstructor.samples())
    {
        for(const auto& rs : vec)
        {
            max_frames_seen = std::max(max_frames_seen, rs.frames.size());
        }
    }
    EXPECT_GE(max_frames_seen, 2U)
        << "default (auto_best) signal config no longer recovers multi-frame "
           "chains via DWARF";
}

TEST(SignalSamplerTest, NoLostUnderLightLoad)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    // Modest frequency + deep ring + frequent drains: overflow must not happen.
    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
    cpu_snap::sampling::sampler smp{ make_signal_cfg(500, std::ref(sink)) };
    smp.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 300 });
    smp.stop();

    stop.store(true);
    worker.join();

    EXPECT_GT(smp.total_samples(), 0U);
    EXPECT_EQ(smp.lost_samples(), 0U);
}

// Lifecycle: the signal backend must be usable repeatedly in one process and in
// any order relative to perf. Regression for the timer_create EAGAIN exhaustion
// caused by timer expirations leaking into a blocked thread's pending set.

// Repeated create/run/destroy over a busy thread. Each iteration must collect
// samples; no lifecycle may fail to arm its timers.
TEST(SignalSamplerLifecycleTest, RepeatedLifecyclesCollectSamples)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    constexpr int k_iterations = 8;
    for(int i = 0; i < k_iterations; ++i)
    {
        std::uint64_t total = 0;
        ASSERT_NO_THROW(
            total = run_signal_lifecycle(1000, std::chrono::milliseconds{ 80 }))
            << "lifecycle " << i << " failed to start";
        EXPECT_GT(total, 0U) << "lifecycle " << i << " collected no samples";
    }

    stop.store(true);
    worker.join();
}

// Runs in a forked child so a failure (child killed by the signal) does not
// take down the whole test binary.
TEST(SignalSamplerLifecycleTest, SecondSamplerWithDifferentOffsetSurvivesAfterFirstDestroyed)
{
    const pid_t pid = ::fork();
    ASSERT_NE(pid, -1) << "fork failed";

    if(pid == 0)
    {
        std::atomic<bool> stop{ false };
        std::thread       worker(spin_worker, std::ref(stop));

        cpu_snap::core::memory_sink sink_a(
            cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
        auto cfg_a                = make_signal_cfg(1000, std::ref(sink_a));
        cfg_a.signal.signal_rt_offset = 0;
        {
            cpu_snap::sampling::sampler smp{ cfg_a };
            smp.start();
            std::this_thread::sleep_for(std::chrono::milliseconds{ 80 });
            smp.stop();
        }

        cpu_snap::core::memory_sink sink_b(
            cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
        auto cfg_b                = make_signal_cfg(1000, std::ref(sink_b));
        cfg_b.signal.signal_rt_offset = 1;
        {
            cpu_snap::sampling::sampler smp{ cfg_b };
            smp.start();
            std::this_thread::sleep_for(std::chrono::milliseconds{ 80 });
            smp.stop();
        }

        stop.store(true);
        worker.join();
        std::_Exit(0);
    }

    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status))
        << "child did not exit normally, status=" << status;
    if(WIFEXITED(status))
    {
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
    if(WIFSIGNALED(status))
    {
        FAIL() << "child killed by signal " << WTERMSIG(status);
    }
}

// A perf-backend run preceding a signal-backend run in the same process must not
// break the signal backend. The perf half is skipped where perf_event_open is
// not permitted; the signal half always runs.
TEST(SignalSamplerLifecycleTest, PerfThenSignalInSameProcess)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    bool perf_ran = false;
    try
    {
        cpu_snap::core::memory_sink perf_sink(
            cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
        cpu_snap::sampling::sampler perf{ make_perf_cfg(1000, std::ref(perf_sink)) };
        perf.start();
        std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
        perf.stop();
        perf_ran = true;
    }
    catch(const std::exception&)
    {
        // perf_event_open denied (paranoid/container): skip the perf half only.
    }

    std::uint64_t total = 0;
    ASSERT_NO_THROW(
        total = run_signal_lifecycle(1000, std::chrono::milliseconds{ 150 }))
        << "signal backend failed after perf run (perf_ran=" << perf_ran << ")";
    EXPECT_GT(total, 0U);

    stop.store(true);
    worker.join();
}

// Direct regression for the leak: with a low RLIMIT_SIGPENDING and a worker that
// BLOCKS the sampling signal (so every expiration queues in its private pending
// set and is never delivered), many lifecycles must NOT exhaust the sigqueue.
// Before the teardown flush this threw timer_create EAGAIN within a few cycles.
TEST(SignalSamplerLifecycleTest, BlockedThreadLifecyclesDoNotExhaustSigqueue)
{
    rlimit orig{};
    ASSERT_EQ(::getrlimit(RLIMIT_SIGPENDING, &orig), 0);

    // Cap pending signals low enough that an undrained per-lifecycle leak hits
    // the limit within a handful of cycles, but high enough for one lifecycle's
    // live working set (timers + transient pending) to fit.
    rlimit capped = orig;
    constexpr rlim_t k_cap = 40;
    if(orig.rlim_cur == RLIM_INFINITY || orig.rlim_cur > k_cap)
    {
        capped.rlim_cur = k_cap;
    }
    if(::setrlimit(RLIMIT_SIGPENDING, &capped) != 0)
    {
        GTEST_SKIP() << "cannot lower RLIMIT_SIGPENDING";
    }

    std::atomic<bool> stop{ false };
    std::atomic<int>  ready{ 0 };
    constexpr int     k_blockers = 6;
    std::vector<std::thread> blockers;
    blockers.reserve(k_blockers);
    for(int i = 0; i < k_blockers; ++i)
    {
        blockers.emplace_back([&] {
            sigset_t set;
            ::sigemptyset(&set);
            ::sigaddset(&set, k_exhaustion_signal_no);
            ::pthread_sigmask(SIG_BLOCK, &set, nullptr);
            ready.fetch_add(1, std::memory_order_release);
            double acc = 0.0;
            while(!stop.load(std::memory_order_relaxed))
            {
                acc += busy_outer(acc, 300'000);
            }
            if(acc == 12345.6789) std::printf("unreachable %f\n", acc);
        });
    }
    while(ready.load(std::memory_order_acquire) < k_blockers)
    {
        std::this_thread::yield();
    }

    bool threw = false;
    constexpr int k_iterations = 20;  // ~3x the pre-fix failure point
    for(int i = 0; i < k_iterations && !threw; ++i)
    {
        try
        {
            cpu_snap::core::memory_sink sink(
                cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
            cpu_snap::sampling::sampler smp{ make_aggressive_signal_cfg(10000, std::ref(sink)) };
            smp.start();
            std::this_thread::sleep_for(std::chrono::milliseconds{ 25 });
            smp.stop();
        }
        catch(const std::exception& e)
        {
            threw = true;
            ADD_FAILURE() << "lifecycle " << i << " threw: " << e.what();
        }
    }

    stop.store(true);
    for(auto& t : blockers)
    {
        t.join();
    }
    ::setrlimit(RLIMIT_SIGPENDING, &orig);

    EXPECT_FALSE(threw) << "signal backend exhausted the sigqueue across "
                           "lifecycles (pending-signal leak)";
}

// Direct backend tests: signal-handler hygiene, install/restore, teardown.

namespace
{

using signal_backend = cpu_snap::sampling::triggers::signal::backend;

const int k_backend_signal_no = SIGRTMIN + 10;

signal_backend::config
make_backend_cfg(std::uint32_t freq_hz = 5000, std::size_t ring_capacity = 64,
                 std::uint32_t stack_bytes = 4096)
{
    return signal_backend::config{ .freq_hz              = freq_hz,
                                   .signal_no            = k_backend_signal_no,
                                   .stack_snapshot_bytes = stack_bytes,
                                   .callchain_depth      = 64,
                                   .ring_capacity        = ring_capacity,
                                   .start_delay_ns       = 0 };
}

signal_backend::config
make_cpu_time_cfg(std::uint32_t freq_hz = 2000)
{
    auto cfg    = make_backend_cfg(freq_hz);
    cfg.cpu_time = true;
    return cfg;
}

pid_t
self_tid() noexcept
{
    return static_cast<pid_t>(::syscall(SYS_gettid));
}

// Runs @p body in a forked child and returns its exit code (or -signo if the
// child was killed), isolating process-global state from the test binary.
template <typename Body>
int
run_in_child(Body&& body)
{
    const pid_t pid = ::fork();
    if(pid < 0)
    {
        return -1000;
    }
    if(pid == 0)
    {
        std::_Exit(body());
    }
    int status = 0;
    if(::waitpid(pid, &status, 0) != pid)
    {
        return -1001;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
}

std::size_t
vm_size_bytes()
{
    std::ifstream statm("/proc/self/statm");
    std::size_t   pages = 0;
    statm >> pages;
    return pages * static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
}

}  // namespace

// The handler's own failing syscalls must not leak into the interrupted code's
// errno. RLIMIT_AS is clamped so the handler's alternate-stack mmap fails with
// ENOMEM on every delivery.
TEST(SignalBackendHandlerTest, ErrnoIsPreservedAcrossHandler)
{
    const int code = run_in_child([] {
        signal_backend    backend{ make_backend_cfg() };
        std::atomic<pid_t> target_tid{ 0 };
        std::atomic<bool>  go{ false };
        std::uint64_t      corrupted = 0;
        std::thread        target([&] {
            target_tid.store(self_tid());
            while(!go.load())
            {
                std::this_thread::yield();
            }
            constexpr int k_sentinel = 12345;
            const auto    deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds{ 300 };
            while(std::chrono::steady_clock::now() < deadline)
            {
                errno = k_sentinel;
                for(int i = 0; i < 2000; ++i)
                {
                    asm volatile("" ::: "memory");
                }
                if(errno != k_sentinel)
                {
                    ++corrupted;
                }
            }
        });
        while(target_tid.load() == 0)
        {
            std::this_thread::yield();
        }
        backend.attach_tid(target_tid.load());
        backend.disable_all();

        rlimit orig{};
        ::getrlimit(RLIMIT_AS, &orig);
        rlimit capped   = orig;
        capped.rlim_cur = vm_size_bytes() + 16U * 1024U;
        if(::setrlimit(RLIMIT_AS, &capped) != 0)
        {
            go.store(true);
            target.join();
            return 3;
        }
        backend.enable_all();
        go.store(true);
        target.join();
        ::setrlimit(RLIMIT_AS, &orig);
        backend.disable_all();
        if(backend.wait(0) != 1)
        {
            return 2;
        }
        return corrupted == 0 ? 0 : 1;
    });
    EXPECT_EQ(code, 0) << "1 = errno clobbered by the handler, 2 = handler never ran, 3 = setup";
}

namespace
{

void
spin_for(std::chrono::milliseconds duration)
{
    const auto deadline = std::chrono::steady_clock::now() + duration;
    double     acc      = 0.0;
    while(std::chrono::steady_clock::now() < deadline)
    {
        acc += busy_inner(acc, 20'000);
    }
    if(acc == 12345.6789)
    {
        std::printf("unreachable %f\n", acc);
    }
}

}  // namespace

namespace
{

// A thread that already has an alternate stack (Go/Rust/sanitizer hosts) must
// keep it: the handler may only install its own when none is registered.
void
expect_host_altstack_survives(int host_flags)
{
    constexpr std::size_t k_host_stack_bytes = 128U * 1024U;
    void* host_stack = ::mmap(nullptr, k_host_stack_bytes, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(host_stack, MAP_FAILED);

    signal_backend    backend{ make_backend_cfg() };
    std::atomic<bool> attached{ false };
    std::atomic<pid_t> tid{ 0 };
    void*             seen_sp = nullptr;
    std::thread       target([&] {
        stack_t ss{};
        ss.ss_sp    = host_stack;
        ss.ss_size  = k_host_stack_bytes;
        ss.ss_flags = host_flags;
        ::sigaltstack(&ss, nullptr);
        tid.store(self_tid());
        while(!attached.load())
        {
            std::this_thread::yield();
        }
        spin_for(std::chrono::milliseconds{ 100 });
        stack_t cur{};
        ::sigaltstack(nullptr, &cur);
        seen_sp = cur.ss_sp;
    });
    while(tid.load() == 0)
    {
        std::this_thread::yield();
    }
    backend.attach_tid(tid.load());
    attached.store(true);
    target.join();
    backend.close_all();

    EXPECT_EQ(seen_sp, host_stack);
    ::munmap(host_stack, k_host_stack_bytes);
}

}  // namespace

TEST(SignalBackendAltstackTest, KeepsExistingAlternateStack)
{
    expect_host_altstack_survives(0);
}

// SS_AUTODISARM lets sigaltstack succeed from inside a handler running on the
// stack, which is the case where an unconditional install would overwrite it.
TEST(SignalBackendAltstackTest, KeepsExistingAutodisarmAlternateStack)
{
    expect_host_altstack_survives(static_cast<int>(1U << 31));  // SS_AUTODISARM
}

// The handler installs an alternate stack on the first delivery; it must still
// be registered after the handler returns (sigreturn restores the stack saved
// in the signal frame, so a plain sigaltstack() inside the handler is undone).
TEST(SignalBackendAltstackTest, InstalledAlternateStackSurvivesHandlerReturn)
{
    signal_backend     backend{ make_backend_cfg() };
    std::atomic<bool>  attached{ false };
    std::atomic<pid_t> tid{ 0 };
    stack_t            after{};
    std::thread        target([&] {
        tid.store(self_tid());
        while(!attached.load())
        {
            std::this_thread::yield();
        }
        spin_for(std::chrono::milliseconds{ 100 });
        ::sigaltstack(nullptr, &after);
    });
    while(tid.load() == 0)
    {
        std::this_thread::yield();
    }
    backend.attach_tid(tid.load());
    attached.store(true);
    target.join();
    backend.close_all();

    EXPECT_EQ(after.ss_flags & SS_DISABLE, 0) << "thread was left without an alternate stack";
    EXPECT_NE(after.ss_sp, nullptr);
}

// The handler-installed alternate stack of a thread that exited must be
// returned once the worker detaches that thread.
TEST(SignalBackendAltstackTest, ReclaimsAlternateStacksOfExitedThreads)
{
    signal_backend backend{ make_backend_cfg(5000, 4, 1024) };

    const auto run_one_thread = [&] {
        std::atomic<pid_t> tid{ 0 };
        std::atomic<bool>  stop{ false };
        std::thread        target([&] {
            tid.store(self_tid());
            while(!stop.load())
            {
                spin_for(std::chrono::milliseconds{ 1 });
            }
        });
        while(tid.load() == 0)
        {
            std::this_thread::yield();
        }
        backend.attach_tid(tid.load());
        std::this_thread::sleep_for(std::chrono::milliseconds{ 20 });
        stop.store(true);
        target.join();
        backend.detach_tid(tid.load());
    };

    for(int i = 0; i < 4; ++i)
    {
        run_one_thread();
    }
    const std::size_t before = vm_size_bytes();
    constexpr int     k_threads = 48;
    for(int i = 0; i < k_threads; ++i)
    {
        run_one_thread();
    }
    const std::size_t grown = vm_size_bytes() - std::min(before, vm_size_bytes());

    constexpr std::size_t k_altstack_bytes = 64U * 1024U;
    EXPECT_LT(grown, (k_threads * k_altstack_bytes) / 4)
        << "alternate stacks leaked: " << grown << " bytes of address space gained";
}

namespace
{

void
host_signal_handler(int /*signo*/) noexcept
{
}

struct sigaction
current_action(int signal_no)
{
    struct sigaction cur
    {};
    ::sigaction(signal_no, nullptr, &cur);
    return cur;
}

}  // namespace

TEST(SignalBackendInstallTest, RefusesSignalAlreadyOwnedByHost)
{
    struct sigaction host
    {};
    host.sa_handler = host_signal_handler;
    ::sigemptyset(&host.sa_mask);
    struct sigaction before
    {};
    ASSERT_EQ(::sigaction(k_backend_signal_no, &host, &before), 0);

    EXPECT_THROW(signal_backend{ make_backend_cfg() }, cpu_snap::environment_error);
    EXPECT_EQ(current_action(k_backend_signal_no).sa_handler, &host_signal_handler);

    ::sigaction(k_backend_signal_no, &before, nullptr);
}

TEST(SignalBackendInstallTest, SecondActiveBackendIsUsageError)
{
    signal_backend first{ make_backend_cfg() };
    try
    {
        signal_backend second{ make_backend_cfg() };
        FAIL() << "expected cpu_snap::usage_error";
    }
    catch(const cpu_snap::usage_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("already active"), std::string::npos);
    }
}

TEST(SignalBackendInstallTest, RestoresPreviousActionOnDestruction)
{
    ASSERT_EQ(current_action(k_backend_signal_no).sa_handler, SIG_DFL);
    {
        signal_backend backend{ make_backend_cfg() };
        EXPECT_NE(current_action(k_backend_signal_no).sa_handler, SIG_DFL);
    }
    EXPECT_EQ(current_action(k_backend_signal_no).sa_handler, SIG_DFL);
}

TEST(SignalBackendInstallTest, KeepsCaptureHandlerInstalledAcrossCloseAll)
{
    signal_backend backend{ make_backend_cfg() };
    backend.attach_tid(self_tid());
    backend.close_all();
    EXPECT_NE(current_action(k_backend_signal_no).sa_handler, SIG_DFL);
    EXPECT_NE(current_action(k_backend_signal_no).sa_handler, SIG_IGN);
}

// ring_table: the handler-readable tid -> ring registry.

namespace
{

using cpu_snap::sampling::triggers::signal::ring_table;
using cpu_snap::sampling::triggers::signal::snapshot_ring;

std::vector<pid_t>
tids_sharing_home_slot(std::size_t count)
{
    std::vector<pid_t> tids;
    const std::size_t  home = ring_table::home_slot(1);
    for(pid_t tid = 1; tids.size() < count; ++tid)
    {
        if(ring_table::home_slot(tid) == home)
        {
            tids.push_back(tid);
        }
    }
    return tids;
}

}  // namespace

TEST(SignalRingTableTest, PublishLookupRetireRoundTrip)
{
    ring_table    table;
    snapshot_ring ring{ 2, 8 };

    EXPECT_EQ(table.lookup(100), nullptr);
    ASSERT_TRUE(table.publish(100, &ring));
    EXPECT_EQ(table.lookup(100), &ring);
    EXPECT_NE(table.retire(100), ring_table::k_npos);
    EXPECT_EQ(table.lookup(100), nullptr);
    EXPECT_EQ(table.retire(100), ring_table::k_npos);
}

TEST(SignalRingTableTest, PublishFailsWhenProbeWindowIsFull)
{
    ring_table    table;
    snapshot_ring ring{ 2, 8 };
    const auto    tids = tids_sharing_home_slot(ring_table::k_max_probe + 1);

    for(std::size_t i = 0; i < ring_table::k_max_probe; ++i)
    {
        ASSERT_TRUE(table.publish(tids[i], &ring)) << "entry " << i;
    }
    EXPECT_FALSE(table.publish(tids.back(), &ring));
    EXPECT_EQ(table.lookup(tids.back()), nullptr);

    EXPECT_NE(table.retire(tids.front()), ring_table::k_npos);
    EXPECT_TRUE(table.publish(tids.back(), &ring));
    EXPECT_EQ(table.lookup(tids.back()), &ring);
}

TEST(SignalRingTableTest, ChurnDoesNotAccumulateTombstones)
{
    ring_table    table;
    snapshot_ring ring{ 2, 8 };

    for(pid_t tid = 1; tid <= 20000; ++tid)
    {
        ASSERT_TRUE(table.publish(tid, &ring));
        ASSERT_NE(table.retire(tid), ring_table::k_npos);
    }
    EXPECT_EQ(table.occupied_slots(), 0U);
}

TEST(SignalRingTableTest, RetiredSlotIsNotIdleWhileAHandlerIsInside)
{
    ring_table    table;
    snapshot_ring ring{ 2, 8 };
    ASSERT_TRUE(table.publish(100, &ring));

    std::size_t    slot    = ring_table::k_npos;
    snapshot_ring* entered = table.enter(100, slot);
    ASSERT_EQ(entered, &ring);

    const std::size_t retired = table.retire(100);
    ASSERT_EQ(retired, slot);
    EXPECT_FALSE(table.idle(retired));
    EXPECT_EQ(table.enter(100, slot), nullptr);

    table.leave(retired);
    EXPECT_TRUE(table.idle(retired));
}

TEST(SignalRingTableTest, EntriesBehindATombstoneStayReachable)
{
    ring_table    table;
    snapshot_ring ring{ 2, 8 };
    const auto    tids = tids_sharing_home_slot(3);

    ASSERT_TRUE(table.publish(tids[0], &ring));
    ASSERT_TRUE(table.publish(tids[1], &ring));
    ASSERT_TRUE(table.publish(tids[2], &ring));
    EXPECT_NE(table.retire(tids[0]), ring_table::k_npos);
    EXPECT_EQ(table.lookup(tids[1]), &ring);
    EXPECT_EQ(table.lookup(tids[2]), &ring);
}

// More live threads than the table can hold: the overflow must fail the attach
// instead of arming a timer whose samples the handler would silently drop.
TEST(SignalBackendTableTest, AttachBeyondTableCapacityThrowsAndLeavesNothingArmed)
{
    signal_backend backend{ make_backend_cfg(1, 2, 8) };

    constexpr int           k_threads = static_cast<int>(ring_table::k_slots) + 64;
    std::atomic<bool>       stop{ false };
    std::atomic<int>        started{ 0 };
    std::vector<pid_t>      tids(k_threads, 0);
    std::vector<std::thread> threads;
    threads.reserve(k_threads);
    for(int i = 0; i < k_threads; ++i)
    {
        threads.emplace_back([&, i] {
            tids[static_cast<std::size_t>(i)] = self_tid();
            started.fetch_add(1);
            while(!stop.load())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds{ 5 });
            }
        });
    }
    while(started.load() < k_threads)
    {
        std::this_thread::yield();
    }

    int failed = 0;
    for(const pid_t tid : tids)
    {
        try
        {
            backend.attach_tid(tid);
            EXPECT_TRUE(backend.is_tracked(tid));
        }
        catch(const cpu_snap::error&)
        {
            ++failed;
            EXPECT_FALSE(backend.is_tracked(tid));
        }
    }
    EXPECT_GE(failed, 64);
    EXPECT_EQ(backend.tracked_tids(), static_cast<std::size_t>(k_threads - failed));

    backend.close_all();
    stop.store(true);
    for(auto& t : threads)
    {
        t.join();
    }
}

// Rings of detached threads must be released once no handler can still be
// writing to them, not parked until the backend is destroyed.
TEST(SignalBackendTableTest, DetachedRingsAreFreedBeforeBackendDestruction)
{
    constexpr std::size_t k_ring_slots = 256;
    constexpr std::uint32_t k_stack_bytes = 8192;
    signal_backend backend{ make_backend_cfg(100, k_ring_slots, k_stack_bytes) };

    const auto run_one_thread = [&] {
        std::atomic<pid_t> tid{ 0 };
        std::atomic<bool>  stop{ false };
        std::thread        target([&] {
            tid.store(self_tid());
            while(!stop.load())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds{ 1 });
            }
        });
        while(tid.load() == 0)
        {
            std::this_thread::yield();
        }
        backend.attach_tid(tid.load());
        stop.store(true);
        target.join();
        backend.detach_tid(tid.load());
    };

    run_one_thread();
    const std::size_t before = ::mallinfo2().uordblks;
    constexpr int     k_threads = 24;
    for(int i = 0; i < k_threads; ++i)
    {
        run_one_thread();
    }
    const std::size_t after = ::mallinfo2().uordblks;

    constexpr std::size_t k_ring_bytes = k_ring_slots * k_stack_bytes;
    EXPECT_LT(after - std::min(before, after), 2 * k_ring_bytes)
        << "retired rings are accumulating (" << k_threads << " rings of " << k_ring_bytes
        << " bytes)";
}

// Detaching and re-attaching a live, sampled thread while its handler keeps
// firing exercises ring retirement against in-flight handlers (meaningful under
// ASan/TSan).
TEST(SignalBackendTableTest, DetachAttachChurnOnSampledThreadIsSafe)
{
    signal_backend     backend{ make_backend_cfg(20000, 16, 1024) };
    std::atomic<bool>  stop{ false };
    std::atomic<pid_t> tid{ 0 };
    std::thread        target([&] {
        tid.store(self_tid());
        while(!stop.load())
        {
            spin_for(std::chrono::milliseconds{ 1 });
        }
    });
    while(tid.load() == 0)
    {
        std::this_thread::yield();
    }
    for(int i = 0; i < 200; ++i)
    {
        backend.attach_tid(tid.load());
        std::this_thread::sleep_for(std::chrono::microseconds{ 500 });
        backend.detach_tid(tid.load());
    }
    stop.store(true);
    target.join();
    EXPECT_EQ(backend.tracked_tids(), 0U);
}

// Lifecycle: restart after stop(), fork.

TEST(SignalSamplerLifecycleTest, RestartAfterStopCollectsSamplesAgain)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
    cpu_snap::sampling::sampler smp{ make_signal_cfg(1000, std::ref(sink)) };

    smp.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
    smp.stop();
    const std::uint64_t first_run = smp.total_samples();
    ASSERT_GT(first_run, 0U);

    smp.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
    smp.stop();
    EXPECT_GT(smp.total_samples(), first_run) << "second start() produced no samples";

    stop.store(true);
    worker.join();
}

// A sampler created in a forked child must work even though the parent's
// backend (copied into the child) and the parent's cached thread id are stale.
TEST(SignalSamplerLifecycleTest, SamplerCreatedInForkedChildCollectsSamples)
{
    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_test_sink_max_bytes });
    cpu_snap::sampling::sampler parent{ make_signal_cfg(1000, std::ref(sink)) };
    parent.start();
    spin_for(std::chrono::milliseconds{ 100 });

    const int code = run_in_child([] {
        try
        {
            cpu_snap::core::memory_sink child_sink(cpu_snap::core::memory_sink::config{
                .max_bytes = k_test_sink_max_bytes });
            auto cfg = make_signal_cfg(1000, std::ref(child_sink));
            cfg.threads.include_main_thread = true;
            cpu_snap::sampling::sampler child{ std::move(cfg) };
            child.start();
            spin_for(std::chrono::milliseconds{ 150 });
            child.stop();
            return child.total_samples() > 0 ? 0 : 2;
        }
        catch(const std::exception&)
        {
            return 1;
        }
    });
    EXPECT_EQ(code, 0) << "1 = sampler construction/start threw, 2 = no samples";

    parent.stop();
    stop.store(true);
    worker.join();
}

// The handler must not post the wake eventfd for every sample: a quarter-full
// ring is the wake point, the worker's poll timeout covers everything below it.
TEST(SignalBackendWakeTest, WakesOnlyWhenRingCrossesFillThreshold)
{
    signal_backend     backend{ make_backend_cfg(2000, 4096, 1024) };
    std::atomic<bool>  stop{ false };
    std::atomic<pid_t> tid{ 0 };
    std::thread        target([&] {
        tid.store(self_tid());
        while(!stop.load())
        {
            spin_for(std::chrono::milliseconds{ 1 });
        }
    });
    while(tid.load() == 0)
    {
        std::this_thread::yield();
    }
    backend.attach_tid(tid.load());

    std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
    EXPECT_EQ(backend.wait(0), 0) << "worker was woken for a sparsely filled ring";
    EXPECT_EQ(backend.wait(5000), 1) << "worker was never woken as the ring filled";

    backend.close_all();
    stop.store(true);
    target.join();
}

// Wall-clock timers also expire on sleeping threads and interrupt their
// blocking syscalls with EINTR; the CPU-time clock only ticks while the sampled
// thread runs.
namespace
{

int
count_eintr_while_sleeping(const signal_backend::config& cfg)
{
    signal_backend     backend{ cfg };
    std::atomic<pid_t> tid{ 0 };
    std::atomic<bool>  done{ false };
    std::atomic<int>   interrupted{ 0 };
    std::thread        sleeper([&] {
        tid.store(self_tid());
        const timespec nap{ 0, 20'000'000 };
        for(int i = 0; i < 10; ++i)
        {
            if(::nanosleep(&nap, nullptr) != 0 && errno == EINTR)
            {
                interrupted.fetch_add(1);
            }
        }
        done.store(true);
    });
    while(tid.load() == 0)
    {
        std::this_thread::yield();
    }
    backend.attach_tid(tid.load());
    sleeper.join();
    backend.close_all();
    return interrupted.load();
}

}  // namespace

TEST(SignalBackendClockTest, WallClockTimerInterruptsSleepingThread)
{
    EXPECT_GT(count_eintr_while_sleeping(make_backend_cfg(2000)), 0);
}

TEST(SignalBackendClockTest, CpuTimeTimerLeavesSleepingThreadAlone)
{
    EXPECT_EQ(count_eintr_while_sleeping(make_cpu_time_cfg()), 0);
}

TEST(SignalBackendClockTest, CpuTimeTimerSamplesBusyThread)
{
    signal_backend     backend{ make_cpu_time_cfg() };
    std::atomic<bool>  stop{ false };
    std::atomic<pid_t> tid{ 0 };
    std::thread        target([&] {
        tid.store(self_tid());
        while(!stop.load())
        {
            spin_for(std::chrono::milliseconds{ 1 });
        }
    });
    while(tid.load() == 0)
    {
        std::this_thread::yield();
    }
    backend.attach_tid(tid.load());
    EXPECT_EQ(backend.wait(5000), 1) << "no sample from a busy thread under the CPU-time clock";

    backend.close_all();
    stop.store(true);
    target.join();
}

TEST(SignalBackendClockTest, CpuTimeAttachToExitedThreadIsSilent)
{
    signal_backend     backend{ make_cpu_time_cfg() };
    std::atomic<pid_t> tid{ 0 };
    std::thread        brief([&] { tid.store(self_tid()); });
    brief.join();
    EXPECT_NO_THROW(backend.attach_tid(tid.load()));
    EXPECT_FALSE(backend.is_tracked(tid.load()));
}
