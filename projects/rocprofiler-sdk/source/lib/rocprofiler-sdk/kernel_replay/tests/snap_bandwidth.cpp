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

// Performance regression tests for kernel-replay snap/restore. The design cost model
// (kernel-replay-arch.tex Figure 5) assumes 45-55 GB/s host-link bandwidth on gfx942 and
// P x footprint bytes moved per replay window. These tests measure effective snap+restore
// bandwidth on the live tracker inventory and fail if throughput collapses or wall time
// exceeds a conservative multiple of the model.

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

size_t
snapshot_footprint_bytes(const msnp::device_snapshot_t& snap)
{
    size_t total = 0;
    for(const auto& block : snap.blocks) total += block.host_copy.size();
    return total;
}

// Conservative host-link floor (GB/s). Figure 5 cites 45-55 GB/s on gfx942; CI runs on
// heterogeneous GPUs so the floor is intentionally low. Override with
// ROCPROFILER_KR_MIN_SNAP_GBPS for tighter local checks.
double
min_snap_bandwidth_gbps()
{
    if(const char* env = std::getenv("ROCPROFILER_KR_MIN_SNAP_GBPS"))
    {
        return std::atof(env);
    }
    return 8.0;
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
    double snap_seconds     = 0.0;
    double restore_seconds  = 0.0;
    size_t footprint_bytes  = 0;
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
    out.snap_seconds =
        std::chrono::duration<double>(snap_end - snap_start).count();
    out.restore_seconds =
        std::chrono::duration<double>(restore_end - restore_start).count();
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
        t.restore_seconds > 0.0
            ? static_cast<double>(t.footprint_bytes) / t.restore_seconds / 1e9
            : 0.0;
    const double snap_gbps =
        t.snap_seconds > 0.0
            ? static_cast<double>(t.footprint_bytes) / t.snap_seconds / 1e9
            : 0.0;
    // Emitted for CI artifact collection and kernel-replay-arch.tex updates.
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
void
run_bandwidth_test(size_t ballast_bytes, const char* label)
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
    {
        const auto warmup = measure_snap_restore_once(agent, buffer, n_elems);
        ASSERT_GT(warmup.footprint_bytes, 0U) << "snap/restore warmup failed";
    }

    constexpr int kIterations = 3;
    const auto    timing =
        measure_snap_restore_mean(agent, buffer, n_elems, kIterations);
    ASSERT_GT(timing.footprint_bytes, 0U) << "snap/restore measurement failed";
    log_timing(label, timing);

    const double min_restore_gbps = min_snap_bandwidth_gbps();
    const double restore_gbps =
        timing.restore_seconds > 0.0
            ? static_cast<double>(timing.footprint_bytes) / timing.restore_seconds / 1e9
            : 0.0;

    EXPECT_GE(restore_gbps, min_restore_gbps)
        << label << ": restore bandwidth " << restore_gbps << " GB/s below floor "
        << min_restore_gbps << " GB/s (footprint " << timing.footprint_bytes << " bytes)";

    // Absolute wall-time guards (catch blow-ups without assuming snap is pure DMA).
    const double total_seconds = timing.snap_seconds + timing.restore_seconds;
    const double ballast_mb =
        static_cast<double>(timing.footprint_bytes) / (1024.0 * 1024.0);
    const double max_total_seconds =
        (static_cast<double>(timing.footprint_bytes) / (min_restore_gbps * 1e9))
            * max_wall_time_margin()
        + 0.10 + ballast_mb * 0.002;  // snap inventory/module-var discovery allowance
    EXPECT_LE(total_seconds, max_total_seconds)
        << label << ": snap+restore wall time " << total_seconds << " s exceeds ceiling "
        << max_total_seconds << " s";

    ASSERT_EQ(hipFree(buffer), hipSuccess);
}
}  // namespace

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
        << "snap+restore scaled super-linearly: small=" << small_total << " s large="
        << large_total << " s ratio=" << (large_total / small_total);

    ASSERT_EQ(hipFree(small), hipSuccess);
    ASSERT_EQ(hipFree(large), hipSuccess);
}
