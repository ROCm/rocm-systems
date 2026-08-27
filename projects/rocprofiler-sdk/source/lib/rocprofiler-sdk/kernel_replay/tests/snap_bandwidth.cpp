// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Performance regression tests for kernel-replay snap/restore. Guards against throughput
// collapse and wall-time blow-ups using the Figure 5 cost model (P x footprint bytes
// over the host link). Thresholds are architecture-agnostic — suitable for any AMD GPU.

#include "snap_kernels.hpp"

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace rocprofiler;
namespace mt   = kernel_replay::memory_tracker;
namespace msnp = kernel_replay::memory_snapshot;

namespace
{
rocprofiler_context_id_t g_ctx{};

void
tracing_noop(rocprofiler_callback_tracing_record_t /*record*/,
             rocprofiler_user_data_t* /*user_data*/,
             void* /*callback_data*/)
{}

int
tool_init(rocprofiler_client_finalize_t /*fini*/, void* /*tool_data*/)
{
    if(rocprofiler_create_context(&g_ctx) != ROCPROFILER_STATUS_SUCCESS) return -1;
    rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API, nullptr, 0, tracing_noop, nullptr);
    rocprofiler_start_context(g_ctx);
    return 0;
}

void
tool_fini(void* /*tool_data*/)
{}

rocprofiler_tool_configure_result_t*
configure(uint32_t /*version*/,
          const char* /*runtime_version*/,
          uint32_t /*priority*/,
          rocprofiler_client_id_t* id)
{
    id->name        = "kernel-replay-snap-bandwidth-test";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}

bool
ensure_live_tracking()
{
    static bool ok = [] {
        if(rocprofiler_force_configure(&configure) != ROCPROFILER_STATUS_SUCCESS) return false;
        if(hipInit(0) != hipSuccess) return false;
        int devs = 0;
        if(hipGetDeviceCount(&devs) != hipSuccess || devs == 0) return false;
        return true;
    }();
    if(ok) mt::set_tracking_enabled(true);
    return ok;
}

hsa_agent_t
gpu_agent()
{
    for(const auto* rocp_agent : rocprofiler::agent::get_agents())
    {
        if(rocp_agent == nullptr || rocp_agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;
        if(auto hsa = rocprofiler::agent::get_hsa_agent(rocp_agent); hsa.has_value()) return *hsa;
    }
    return hsa_agent_t{.handle = 0};
}

void
sync_ok()
{
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

std::vector<float>
read_device(const float* d, int n)
{
    std::vector<float> out(static_cast<size_t>(n));
    EXPECT_EQ(
        hipMemcpy(out.data(), d, static_cast<size_t>(n) * sizeof(float), hipMemcpyDeviceToHost),
        hipSuccess);
    return out;
}

size_t
snapshot_footprint_bytes(const msnp::device_snapshot_t& snap)
{
    size_t total = 0;
    for(const auto& block : snap.blocks)
        total += block.host_copy.size();
    return total;
}

// Conservative host-link floor (GB/s). Intentionally low for heterogeneous CI including
// older AMD GPUs. Override with ROCPROFILER_KR_MIN_SNAP_GBPS for tighter local checks.
double
min_snap_bandwidth_gbps()
{
    if(const char* env = std::getenv("ROCPROFILER_KR_MIN_SNAP_GBPS"))
    {
        return std::atof(env);
    }
    return 4.0;
}

// Wall-time ceiling multiplier over the bandwidth-model prediction.
double
max_wall_time_margin()
{
    if(const char* env = std::getenv("ROCPROFILER_KR_WALL_TIME_MARGIN"))
    {
        return std::atof(env);
    }
    return 4.0;
}

struct snap_restore_timing_t
{
    double snap_seconds    = 0.0;
    double restore_seconds = 0.0;
    size_t footprint_bytes = 0;
};

snap_restore_timing_t
measure_snap_restore_once(hsa_agent_t agent, float* buffer, int n_elems)
{
    using clock = std::chrono::steady_clock;

    auto snap_start = clock::now();
    auto snapshot   = msnp::snap(agent);
    auto snap_end   = clock::now();
    if(!snapshot.ok) return {};

    kernel_launch::add(buffer, 1.0f, n_elems);
    sync_ok();

    auto restore_start = clock::now();
    if(!msnp::restore(snapshot)) return {};
    auto restore_end = clock::now();

    snap_restore_timing_t out{};
    out.footprint_bytes = snapshot_footprint_bytes(snapshot);
    out.snap_seconds    = std::chrono::duration<double>(snap_end - snap_start).count();
    out.restore_seconds = std::chrono::duration<double>(restore_end - restore_start).count();
    return out;
}

snap_restore_timing_t
measure_snap_restore_mean(hsa_agent_t agent, float* buffer, int n_elems, int iterations)
{
    snap_restore_timing_t accum{};
    for(int i = 0; i < iterations; ++i)
    {
        auto t = measure_snap_restore_once(agent, buffer, n_elems);
        accum.snap_seconds += t.snap_seconds;
        accum.restore_seconds += t.restore_seconds;
        accum.footprint_bytes = t.footprint_bytes;
    }
    accum.snap_seconds /= static_cast<double>(iterations);
    accum.restore_seconds /= static_cast<double>(iterations);
    return accum;
}

void
log_timing(const char* label, const snap_restore_timing_t& t)
{
    const double restore_gbps =
        t.restore_seconds > 0.0 ? static_cast<double>(t.footprint_bytes) / t.restore_seconds / 1e9
                                : 0.0;
    const double snap_gbps =
        t.snap_seconds > 0.0 ? static_cast<double>(t.footprint_bytes) / t.snap_seconds / 1e9 : 0.0;
    // Emitted for CI log collection (local/paper analysis only — not checked into PRs).
    std::printf("[kr-perf] %s footprint_mb=%.2f snap_ms=%.3f restore_ms=%.3f "
                "restore_gbps=%.2f snap_gbps=%.2f\n",
                label,
                static_cast<double>(t.footprint_bytes) / (1024.0 * 1024.0),
                t.snap_seconds * 1000.0,
                t.restore_seconds * 1000.0,
                restore_gbps,
                snap_gbps);
}

// snap() includes inventory scan and module-variable discovery in addition to DMA; restore()
// is dominated by host->device copies and matches the Figure 5 restore term.
// check_bandwidth_floor: a GB/s floor only means something once the transfer is large enough to
// amortize per-transfer setup. At the smallest footprint the copy is a couple of milliseconds and
// setup dominates, so the computed rate reports fixed cost rather than link bandwidth. Those cases
// still run, and still assert the wall-time ceiling below, which is what catches a blow-up.
void
run_bandwidth_test(size_t      ballast_bytes,
                   const char* label,
                   int         warmup_iterations     = 1,
                   int         measure_iterations    = 3,
                   bool        check_bandwidth_floor = true)
{
    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U);

    const int n_elems = static_cast<int>(ballast_bytes / sizeof(float));
    float*    buffer  = nullptr;
    ASSERT_EQ(hipMalloc(&buffer, ballast_bytes), hipSuccess);
    ASSERT_NE(buffer, nullptr);
    kernel_launch::fill(buffer, 0.0f, n_elems);
    sync_ok();

    // Warmup: prime caches and allocator state before timed iterations.
    for(int w = 0; w < warmup_iterations; ++w)
    {
        const auto warmup = measure_snap_restore_once(agent, buffer, n_elems);
        ASSERT_GT(warmup.footprint_bytes, 0U) << "snap/restore warmup failed";
    }

    const auto timing = measure_snap_restore_mean(agent, buffer, n_elems, measure_iterations);
    ASSERT_GT(timing.footprint_bytes, 0U) << "snap/restore measurement failed";
    log_timing(label, timing);

    const double min_restore_gbps = min_snap_bandwidth_gbps();
    const double restore_gbps =
        timing.restore_seconds > 0.0
            ? static_cast<double>(timing.footprint_bytes) / timing.restore_seconds / 1e9
            : 0.0;

    if(check_bandwidth_floor)
    {
        EXPECT_GE(restore_gbps, min_restore_gbps)
            << label << ": restore bandwidth " << restore_gbps << " GB/s below floor "
            << min_restore_gbps << " GB/s (footprint " << timing.footprint_bytes << " bytes)";
    }

    // Absolute wall-time guards (catch blow-ups without assuming snap is pure DMA).
    const double total_seconds = timing.snap_seconds + timing.restore_seconds;
    const double ballast_mb    = static_cast<double>(timing.footprint_bytes) / (1024.0 * 1024.0);
    const double max_total_seconds =
        (static_cast<double>(timing.footprint_bytes) / (min_restore_gbps * 1e9)) *
            max_wall_time_margin() +
        0.10 + ballast_mb * 0.002;  // snap inventory/module-var discovery allowance
    EXPECT_LE(total_seconds, max_total_seconds)
        << label << ": snap+restore wall time " << total_seconds << " s exceeds ceiling "
        << max_total_seconds << " s";

    ASSERT_EQ(hipFree(buffer), hipSuccess);
}
}  // namespace

TEST(kernel_replay_snapshot, snap_bandwidth_meets_floor_with_8mb_ballast)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";
    // 8 MB restores in ~2 ms, which is not long enough for a GB/s figure to describe the link
    // rather than the per-transfer setup around it, so the bandwidth floor is not asserted here.
    // The wall-time ceiling still applies, and the 32/128 MB cases carry the bandwidth floor.
    run_bandwidth_test(8U * 1024U * 1024U, "ballast_8mb", 3, 5, /*check_bandwidth_floor=*/false);
}

TEST(kernel_replay_snapshot, snap_bandwidth_meets_floor_with_32mb_ballast)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";
    run_bandwidth_test(32U * 1024U * 1024U, "ballast_32mb");
}

TEST(kernel_replay_snapshot, snap_bandwidth_meets_floor_with_128mb_ballast)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";
    run_bandwidth_test(128U * 1024U * 1024U, "ballast_128mb");
}

// Snap/restore cost should grow roughly with footprint (bandwidth-bound), not super-linearly.
TEST(kernel_replay_snapshot, snap_cost_scales_sublinearly_with_footprint)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U);

    constexpr size_t kSmall = 32U * 1024U * 1024U;
    constexpr size_t kLarge = 128U * 1024U * 1024U;

    float* small = nullptr;
    float* large = nullptr;
    ASSERT_EQ(hipMalloc(&small, kSmall), hipSuccess);
    ASSERT_EQ(hipMalloc(&large, kLarge), hipSuccess);

    const int small_elems = static_cast<int>(kSmall / sizeof(float));
    const int large_elems = static_cast<int>(kLarge / sizeof(float));
    kernel_launch::fill(small, 0.0f, small_elems);
    kernel_launch::fill(large, 0.0f, large_elems);
    sync_ok();

    {
        const auto w = measure_snap_restore_once(agent, small, small_elems);
        ASSERT_GT(w.footprint_bytes, 0U);
    }
    {
        const auto w = measure_snap_restore_once(agent, large, large_elems);
        ASSERT_GT(w.footprint_bytes, 0U);
    }

    const auto t_small = measure_snap_restore_mean(agent, small, small_elems, 3);
    const auto t_large = measure_snap_restore_mean(agent, large, large_elems, 3);
    ASSERT_GT(t_small.footprint_bytes, 0U);
    ASSERT_GT(t_large.footprint_bytes, 0U);

    log_timing("scale_small_32mb", t_small);
    log_timing("scale_large_128mb", t_large);

    const double small_total = t_small.snap_seconds + t_small.restore_seconds;
    const double large_total = t_large.snap_seconds + t_large.restore_seconds;
    ASSERT_GT(small_total, 0.0);

    // 4x footprint (32->128 MB ballast) must not blow up faster than 6x wall time.
    // Fixed HIP-runtime inventory adds a constant offset, so the ratio is capped above 4.
    EXPECT_LT(large_total / small_total, 6.0)
        << "snap+restore scaled super-linearly: small=" << small_total << " s large=" << large_total
        << " s ratio=" << (large_total / small_total);

    ASSERT_EQ(hipFree(small), hipSuccess);
    ASSERT_EQ(hipFree(large), hipSuccess);
}

// Consecutive snap+restore cycles should not oscillate wildly (catches allocator thrash).
TEST(kernel_replay_snapshot, snap_restore_wall_time_stable_across_iterations)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto       agent    = gpu_agent();
    constexpr size_t kBallast = 32U * 1024U * 1024U;
    const int        n_elems  = static_cast<int>(kBallast / sizeof(float));
    float*           buffer   = nullptr;
    ASSERT_NE(agent.handle, 0U);
    ASSERT_EQ(hipMalloc(&buffer, kBallast), hipSuccess);
    kernel_launch::fill(buffer, 0.0f, n_elems);
    sync_ok();

    {
        const auto w = measure_snap_restore_once(agent, buffer, n_elems);
        ASSERT_GT(w.footprint_bytes, 0U);
    }

    constexpr int kSamples  = 5;
    double        min_total = 1e9;
    double        max_total = 0.0;
    for(int i = 0; i < kSamples; ++i)
    {
        const auto   t   = measure_snap_restore_once(agent, buffer, n_elems);
        const double tot = t.snap_seconds + t.restore_seconds;
        ASSERT_GT(t.footprint_bytes, 0U);
        min_total = std::min(min_total, tot);
        max_total = std::max(max_total, tot);
    }
    ASSERT_GT(min_total, 0.0);
    // Allow up to 4x spread — enough for CI noise, tight enough to catch blow-ups.
    EXPECT_LT(max_total / min_total, 4.0)
        << "snap+restore wall time unstable: min=" << min_total << " s max=" << max_total
        << " s ratio=" << (max_total / min_total);

    ASSERT_EQ(hipFree(buffer), hipSuccess);
}

// One snapshot must support multiple restores (replay reuses the same image each pass).
TEST(kernel_replay_snapshot, snap_restore_repeatable_from_one_snapshot)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto       agent    = gpu_agent();
    constexpr size_t kBallast = 16U * 1024U * 1024U;
    const int        n_elems  = static_cast<int>(kBallast / sizeof(float));
    float*           buffer   = nullptr;
    ASSERT_NE(agent.handle, 0U);
    ASSERT_EQ(hipMalloc(&buffer, kBallast), hipSuccess);
    kernel_launch::fill(buffer, 1.0f, n_elems);
    sync_ok();

    const auto snap_start = std::chrono::steady_clock::now();
    auto       snapshot   = msnp::snap(agent);
    const auto snap_end   = std::chrono::steady_clock::now();
    ASSERT_TRUE(snapshot.ok);
    ASSERT_GT(snapshot_footprint_bytes(snapshot), 0U);

    // What this test exists to prove is that one snapshot image stays valid across repeated
    // restores, the way a replay loop reuses it every pass. That is a correctness property, so it
    // is checked by reading the buffer back rather than by timing the copy: each restore must undo
    // the kernel's write and leave the originally filled value. Restore bandwidth is covered by the
    // dedicated bandwidth tests above; asserting a floor per restore here gated correctness on
    // runner load, and measured a cold, un-warmed path that never reaches the floor anyway.
    constexpr int   kRestores = 3;
    constexpr float kFilled   = 1.0f;
    for(int i = 0; i < kRestores; ++i)
    {
        kernel_launch::add(buffer, 2.0f, n_elems);
        sync_ok();
        ASSERT_TRUE(msnp::restore(snapshot)) << "restore #" << (i + 1) << " reported failure";
        sync_ok();

        const auto host = read_device(buffer, n_elems);
        ASSERT_EQ(host.size(), static_cast<size_t>(n_elems));
        const auto unrestored =
            std::count_if(host.begin(), host.end(), [](float v) { return v != kFilled; });
        EXPECT_EQ(unrestored, 0) << "restore #" << (i + 1) << " left " << unrestored << " of "
                                 << n_elems << " elements holding the kernel's value instead of "
                                 << kFilled;
    }

    snap_restore_timing_t log{};
    log.footprint_bytes = snapshot_footprint_bytes(snapshot);
    log.snap_seconds    = std::chrono::duration<double>(snap_end - snap_start).count();
    log.restore_seconds = 0.0;
    log_timing("repeat_snapshot_16mb", log);

    ASSERT_EQ(hipFree(buffer), hipSuccess);
}
