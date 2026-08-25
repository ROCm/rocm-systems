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

// Performance tests for the cost term snap_bandwidth.cpp does not cover: the number of regions, as
// distinct from the number of bytes.
//
// snap_bandwidth.cpp measures throughput on a single large buffer, which is the friendly shape. Real
// applications do not have that shape. A framework or a mesh code holds thousands of separate
// allocations, and snap/restore pays a per-region cost on each one: a hash lookup under the tracker
// read lock, a std::vector allocation, and a separate hsa_memory_copy. Because the snapshot
// destination is unpinned host memory, each of those copies also drives a pin / IOMMU-map / DMA /
// unmap cycle rather than a single large transfer, so per-region overhead is not a rounding error.
//
// Bytes are held constant and the region count varied, so any difference measured here is
// per-region cost alone. Thresholds are ratios against the same run's own single-region measurement
// rather than absolute rates, so they hold across GPU generations and on a loaded CI machine.

#include "replay_test_fixture.hpp"
#include "snap_kernels.hpp"

#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace rocprofiler;
using namespace rocprofiler::kernel_replay::test;
namespace mt   = kernel_replay::memory_tracker;
namespace msnp = kernel_replay::memory_snapshot;

namespace
{
// Total bytes held constant across every shape below.
constexpr size_t kTotalBytes = 64U * 1024U * 1024U;

// Ceiling on how much more expensive the many-region shape may be than the one-region shape at
// identical total bytes. Generous on purpose: the point is to catch a per-region cost that grows
// into the dominant term (a lock taken per region, an allocation per region, a synchronous DMA per
// region), not to pin down a specific overhead. Override for a tighter local check.
double
max_region_overhead_ratio()
{
    if(const char* env = std::getenv("ROCPROFILER_KR_MAX_REGION_OVERHEAD"))
    {
        return std::atof(env);
    }
    return 8.0;
}

struct owned_buffers_t
{
    std::vector<float*> ptrs;

    ~owned_buffers_t()
    {
        for(auto* p : ptrs)
            if(p != nullptr) (void) hipFree(p);
    }

    owned_buffers_t()                       = default;
    owned_buffers_t(const owned_buffers_t&) = delete;
    owned_buffers_t(owned_buffers_t&&)      = delete;
    owned_buffers_t& operator=(const owned_buffers_t&) = delete;
    owned_buffers_t& operator=(owned_buffers_t&&) = delete;
};

// Allocate `count` buffers of kTotalBytes/count each and fill them, so the snapshot has to move
// exactly kTotalBytes regardless of the shape.
bool
allocate_shape(owned_buffers_t& out, size_t count)
{
    const size_t per_buffer = kTotalBytes / count;
    const int    elems      = static_cast<int>(per_buffer / sizeof(float));
    if(elems == 0) return false;

    for(size_t i = 0; i < count; ++i)
    {
        float* p = nullptr;
        if(hipMalloc(&p, per_buffer) != hipSuccess || p == nullptr) return false;
        out.ptrs.push_back(p);
        kernel_launch::fill(p, static_cast<float>(i), elems);
    }
    return hipDeviceSynchronize() == hipSuccess;
}

struct timing_t
{
    double snap_seconds    = 0.0;
    double restore_seconds = 0.0;
    size_t bytes           = 0;
    size_t regions         = 0;

    double total_seconds() const { return snap_seconds + restore_seconds; }
};

timing_t
measure(hsa_agent_t agent)
{
    using clock = std::chrono::steady_clock;

    const auto snap_start = clock::now();
    auto       snapshot   = msnp::snap(agent);
    const auto snap_end   = clock::now();
    if(!snapshot.ok) return {};

    const auto restore_start = clock::now();
    const bool restored      = msnp::restore(snapshot);
    const auto restore_end   = clock::now();
    if(!restored) return {};

    auto out = timing_t{};
    for(const auto& block : snapshot.blocks)
        out.bytes += block.host_copy.size();
    out.regions         = snapshot.blocks.size();
    out.snap_seconds    = std::chrono::duration<double>(snap_end - snap_start).count();
    out.restore_seconds = std::chrono::duration<double>(restore_end - restore_start).count();
    return out;
}

// Median of several samples. snap/restore on a shared CI GPU has a long right tail, and a mean lets
// one outlier decide whether the test passes.
timing_t
measure_median(hsa_agent_t agent, int samples)
{
    auto totals = std::vector<double>{};
    auto last   = timing_t{};
    for(int i = 0; i < samples; ++i)
    {
        last = measure(agent);
        if(last.bytes == 0) return {};
        totals.push_back(last.total_seconds());
    }
    std::sort(totals.begin(), totals.end());

    auto out            = last;
    const auto mid      = totals[totals.size() / 2];
    // Report the median as the snap term and fold restore into it; only the total is compared.
    out.snap_seconds    = mid;
    out.restore_seconds = 0.0;
    return out;
}

void
log_shape(const char* label, const timing_t& t)
{
    std::printf("[kr-perf] %s regions=%zu bytes=%zu total_ms=%.3f effective_gbps=%.2f\n",
                label,
                t.regions,
                t.bytes,
                t.total_seconds() * 1000.0,
                t.total_seconds() > 0.0
                    ? static_cast<double>(t.bytes) / t.total_seconds() / 1e9
                    : 0.0);
}
}  // namespace

// The headline test: same bytes, 1 region vs 256 regions.
//
// A regression here means per-region cost has become the dominant term, which shows up in the field
// as replay being unusably slow on any application that holds many allocations -- the common case,
// not an edge case.
TEST(kernel_replay_region_scaling, cost_is_dominated_by_bytes_not_region_count)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    timing_t few{};
    timing_t many{};

    {
        auto buffers = owned_buffers_t{};
        if(!allocate_shape(buffers, 1)) GTEST_SKIP() << "could not allocate the one-region shape";
        (void) measure(agent);  // warm up allocator and host pages
        few = measure_median(agent, 3);
        ASSERT_GT(few.bytes, 0U) << "one-region measurement failed";
        log_shape("regions_1", few);
    }

    {
        auto buffers = owned_buffers_t{};
        if(!allocate_shape(buffers, 256))
            GTEST_SKIP() << "could not allocate the many-region shape";
        (void) measure(agent);
        many = measure_median(agent, 3);
        ASSERT_GT(many.bytes, 0U) << "many-region measurement failed";
        log_shape("regions_256", many);
    }

    // Both shapes must have moved comparable bytes, or the comparison is meaningless. The runtime's
    // own allocations are in both snapshots, so allow a wide band.
    ASSERT_GT(few.bytes, kTotalBytes / 2)
        << "the one-region snapshot did not include the test's buffer";
    ASSERT_GT(many.bytes, kTotalBytes / 2)
        << "the many-region snapshot did not include the test's buffers";

    ASSERT_GT(few.total_seconds(), 0.0);
    const double ratio = many.total_seconds() / few.total_seconds();

    EXPECT_LT(ratio, max_region_overhead_ratio())
        << "splitting the same " << kTotalBytes << " bytes across 256 regions instead of 1 made "
        << "snap+restore " << ratio << "x slower (" << few.total_seconds() * 1000.0 << " ms vs "
        << many.total_seconds() * 1000.0
        << " ms). Per-region cost has become the dominant term, which makes replay unusable for "
           "applications holding many allocations";
}

// Admission control runs on every replayed dispatch and exists to avoid paying for a snapshot. If
// estimating the footprint costs a meaningful fraction of taking the snapshot, the check is a tax
// rather than a saving -- especially on the many-region shape, where it walks the same inventory.
TEST(kernel_replay_region_scaling, footprint_estimate_is_much_cheaper_than_a_snapshot)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    auto buffers = owned_buffers_t{};
    if(!allocate_shape(buffers, 256)) GTEST_SKIP() << "could not allocate the many-region shape";

    using clock = std::chrono::steady_clock;

    (void) msnp::estimate_footprint(agent);  // warm up the executable walk

    constexpr int kEstimates    = 20;
    const auto    estimate_from = clock::now();
    for(int i = 0; i < kEstimates; ++i)
    {
        const auto estimate = msnp::estimate_footprint(agent);
        ASSERT_GT(estimate.bytes, 0U);
    }
    const double estimate_seconds =
        std::chrono::duration<double>(clock::now() - estimate_from).count() / kEstimates;

    const auto snap_from = clock::now();
    const auto snapshot  = msnp::snap(agent);
    const double snap_seconds = std::chrono::duration<double>(clock::now() - snap_from).count();
    ASSERT_TRUE(snapshot.ok);

    std::printf("[kr-perf] estimate_vs_snap estimate_ms=%.4f snap_ms=%.3f\n",
                estimate_seconds * 1000.0,
                snap_seconds * 1000.0);

    ASSERT_GT(snap_seconds, 0.0);
    EXPECT_LT(estimate_seconds, snap_seconds / 10.0)
        << "estimating the footprint took " << estimate_seconds * 1000.0
        << " ms against a snapshot of " << snap_seconds * 1000.0
        << " ms. Admission control is supposed to be the cheap way to avoid the snapshot";
}

// Restore is the term that gets paid P-1 times, so it is the one that decides whether replay is
// affordable. It should not be dramatically more expensive than the single capture, and an asymmetry
// appearing here (host->device markedly slower than device->host at the same region count) is the
// signal that the copy path regressed.
TEST(kernel_replay_region_scaling, restore_is_not_much_slower_than_snap)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    auto buffers = owned_buffers_t{};
    if(!allocate_shape(buffers, 64)) GTEST_SKIP() << "could not allocate buffers";

    (void) measure(agent);

    // Sum a few samples rather than taking one: either direction can be perturbed by an unrelated
    // process on a shared GPU.
    constexpr int kSamples    = 3;
    double        snap_total  = 0.0;
    double        restore_total = 0.0;
    size_t        bytes       = 0;
    for(int i = 0; i < kSamples; ++i)
    {
        const auto t = measure(agent);
        ASSERT_GT(t.bytes, 0U) << "measurement " << i << " failed";
        snap_total += t.snap_seconds;
        restore_total += t.restore_seconds;
        bytes = t.bytes;
    }

    std::printf("[kr-perf] snap_restore_symmetry bytes=%zu snap_ms=%.3f restore_ms=%.3f\n",
                bytes,
                snap_total / kSamples * 1000.0,
                restore_total / kSamples * 1000.0);

    ASSERT_GT(snap_total, 0.0);
    EXPECT_LT(restore_total / snap_total, 4.0)
        << "restore is " << restore_total / snap_total
        << "x the cost of snap. Restore runs once per pass beyond the first, so an asymmetry here "
           "multiplies across the whole replay window";
}

// The generation check added to the restore path runs under the tracker read lock for every region.
// On the many-region shape that is 256 hash lookups plus 256 integer compares per pass, and it must
// not have turned restore into a lock-bound operation.
TEST(kernel_replay_region_scaling, restore_throughput_survives_the_identity_check)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    auto buffers = owned_buffers_t{};
    if(!allocate_shape(buffers, 256)) GTEST_SKIP() << "could not allocate the many-region shape";

    (void) measure(agent);

    const auto snapshot = msnp::snap(agent);
    ASSERT_TRUE(snapshot.ok);

    size_t bytes = 0;
    for(const auto& block : snapshot.blocks)
        bytes += block.host_copy.size();
    ASSERT_GT(bytes, kTotalBytes / 2);

    // Restore the same snapshot repeatedly, as a replay loop does. Every one re-checks identity for
    // every region.
    constexpr int kRestores = 5;
    const auto    from      = std::chrono::steady_clock::now();
    for(int i = 0; i < kRestores; ++i)
        ASSERT_TRUE(msnp::restore(snapshot)) << "restore " << i << " failed";
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count();

    ASSERT_GT(seconds, 0.0);
    const double gbps = static_cast<double>(bytes) * kRestores / seconds / 1e9;

    std::printf("[kr-perf] repeated_restore regions=%zu bytes=%zu restores=%d gbps=%.2f\n",
                snapshot.blocks.size(),
                bytes,
                kRestores,
                gbps);

    // Deliberately far below any link's capability: this is a collapse detector, not a bandwidth
    // assertion. snap_bandwidth.cpp owns the throughput floor on the single-region shape.
    EXPECT_GT(gbps, 0.25)
        << "restoring " << snapshot.blocks.size() << " regions managed only " << gbps
        << " GB/s, which suggests per-region lock or allocation cost now dominates the copy";
}
