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

// Snapshot behaviour under the memory-allocation patterns real applications actually use.
//
// snap_restore.cpp covers the straightforward case: hipMalloc, a kernel writes, restore reverts.
// These tests cover the patterns where that reasoning breaks down, because in each of them the
// snapshot's implicit assumption -- "a device address names one allocation for the whole window,
// and every allocation the kernel touches is in the inventory" -- is false:
//
//   * A caching or pooling allocator frees a buffer and hands the same address back for a different
//     buffer. Deep-learning frameworks, Kokkos, RAPIDS and Thrust all do this by design, so
//     same-address reuse is the common case rather than an unlikely race. (base, size) does not
//     identify an allocation, and restoring on that basis writes a dead buffer's bytes over live
//     data.
//   * A stream-ordered or virtual-memory allocator (hipMallocAsync, hipMemMap, PyTorch
//     expandable_segments, Kokkos with KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC) never calls
//     hsa_amd_memory_pool_allocate, so nothing it allocates reaches the inventory at all.
//   * Managed memory can live on the device and be written by a kernel without being snapshottable.
//
// The oracle for the untracked cases is deliberately not "the allocation must be counted as
// untracked", because whether a given HIP call lands on the VM path depends on the pool
// configuration. It is the invariant that actually matters: every GPU-resident allocation a kernel
// can write must be either snapshottable or counted. An allocation in neither set is invisible --
// the replay reports success having never captured it, and every pass after the first runs on
// mutated inputs with nothing in the output to say so.

#include "replay_test_fixture.hpp"
#include "snap_kernels.hpp"

#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/utils.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

using namespace rocprofiler;
using namespace rocprofiler::kernel_replay::test;
namespace mt   = kernel_replay::memory_tracker;
namespace msnp = kernel_replay::memory_snapshot;

namespace
{
// 4 MB. Large enough that a real DMA happens, small enough to allocate and free repeatedly while
// hunting for address reuse without pressuring a shared CI GPU.
constexpr size_t kBufferBytes = 4U * 1024U * 1024U;
constexpr int    kElems       = static_cast<int>(kBufferBytes / sizeof(float));

std::vector<float>
read_device(const float* d, int n)
{
    std::vector<float> out(n);
    EXPECT_EQ(
        hipMemcpy(out.data(), d, static_cast<size_t>(n) * sizeof(float), hipMemcpyDeviceToHost),
        hipSuccess);
    return out;
}

// True when every element equals `value`. Reported as a bool rather than a sequence of EXPECTs so a
// mismatch does not emit millions of gtest failures.
bool
all_equal(const float* d, int n, float value)
{
    const auto host = read_device(d, n);
    for(int i = 0; i < n; ++i)
        if(host[i] != value) return false;
    return true;
}

bool
in_snapshot_inventory(void* ptr, hsa_agent_t agent)
{
    const auto inv = mt::snap_inventory(agent);
    return inv.find(ptr) != inv.end();
}
}  // namespace

// The generation stamp must distinguish two allocations that occupy the same address at different
// times. Without it, restore() has no way to tell them apart: the map key is the base address and
// the only other recorded property is the size, which a same-size reuse leaves identical.
TEST(kernel_replay_allocator_patterns, generation_distinguishes_reused_address)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    float* first = nullptr;
    ASSERT_EQ(hipMalloc(&first, kBufferBytes), hipSuccess);
    ASSERT_NE(first, nullptr);

    const auto first_generation = mt::generation_of(first);
    EXPECT_NE(first_generation, 0U) << "a tracked allocation must carry a non-zero generation";

    ASSERT_EQ(hipFree(first), hipSuccess);
    EXPECT_EQ(mt::generation_of(first), 0U) << "generation must not survive the free";

    float* second = nullptr;
    ASSERT_EQ(hipMalloc(&second, kBufferBytes), hipSuccess);
    ASSERT_NE(second, nullptr);

    if(second == first)
    {
        EXPECT_NE(mt::generation_of(second), first_generation)
            << "the same address was reused for a new allocation but carries the old generation, "
               "so "
               "restore() cannot tell the two apart";
    }
    else
    {
        // Two live allocations never share a generation regardless of address reuse.
        EXPECT_NE(mt::generation_of(second), first_generation)
            << "two distinct allocations share a generation";
    }

    ASSERT_EQ(hipFree(second), hipSuccess);
}

// The load-bearing case. A buffer is captured, freed, and its address handed to a different buffer
// holding different data; restore() must leave the new buffer alone.
//
// Before the generation check this test fails by writing the old buffer's contents over the new
// one: silent corruption of application data by the profiler, which is the worst failure mode
// available to us. It is reachable in ordinary single-threaded code because the free and the
// reallocation only need to happen between snap() and restore() -- exactly what a caching allocator
// does between two passes of a replay loop.
TEST(kernel_replay_allocator_patterns, restore_does_not_write_into_a_reused_address)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    constexpr float kCaptured = 11.0f;  // contents at snap time
    constexpr float kReused   = 22.0f;  // contents of the allocation that reuses the address

    float* original = nullptr;
    ASSERT_EQ(hipMalloc(&original, kBufferBytes), hipSuccess);
    ASSERT_NE(original, nullptr);
    kernel_launch::fill(original, kCaptured, kElems);
    sync_ok();
    ASSERT_TRUE(in_snapshot_inventory(original, agent)) << "test buffer was not tracked";

    const auto snapshot = msnp::snap(agent);
    ASSERT_TRUE(snapshot.ok);

    ASSERT_EQ(hipFree(original), hipSuccess);

    // Try a few times to land on the same address. HIP's allocator usually returns it immediately
    // for an identical size, but that is not contractual, so the test skips rather than fails when
    // reuse does not happen -- the property under test is unobservable in that case.
    float* reused = nullptr;
    for(int attempt = 0; attempt < 8 && reused != original; ++attempt)
    {
        if(reused != nullptr) ASSERT_EQ(hipFree(reused), hipSuccess);
        reused = nullptr;
        ASSERT_EQ(hipMalloc(&reused, kBufferBytes), hipSuccess);
        ASSERT_NE(reused, nullptr);
    }

    if(reused != original)
    {
        ASSERT_EQ(hipFree(reused), hipSuccess);
        GTEST_SKIP() << "allocator did not reuse the freed address, so address reuse is not "
                        "observable in this run";
    }

    kernel_launch::fill(reused, kReused, kElems);
    sync_ok();
    ASSERT_TRUE(all_equal(reused, kElems, kReused)) << "failed to seed the reusing allocation";

    // restore() must skip this region: the address is live and the right size, but it is a
    // different allocation than the one captured.
    EXPECT_TRUE(msnp::restore(snapshot))
        << "restore must report success -- skipping a replaced region is not a failure";

    EXPECT_TRUE(all_equal(reused, kElems, kReused))
        << "restore wrote the captured allocation's contents into a different allocation that "
           "reused its address; the profiler corrupted application data";

    ASSERT_EQ(hipFree(reused), hipSuccess);
}

// Restoring after a plain free must be a no-op rather than a write to retired device memory. This
// is the pre-existing liveness check; kept here so the generation change cannot silently break it.
TEST(kernel_replay_allocator_patterns, restore_skips_a_freed_region)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    float* buffer = nullptr;
    ASSERT_EQ(hipMalloc(&buffer, kBufferBytes), hipSuccess);
    kernel_launch::fill(buffer, 5.0f, kElems);
    sync_ok();

    const auto snapshot = msnp::snap(agent);
    ASSERT_TRUE(snapshot.ok);

    ASSERT_EQ(hipFree(buffer), hipSuccess);

    EXPECT_TRUE(msnp::restore(snapshot))
        << "a region freed after snap must be skipped, not treated as a restore failure";
}

// Every GPU-resident allocation must be either snapshottable or counted as untracked. An allocation
// in neither set is the silent-wrong-answer case: replay reports success, the kernel writes memory
// nothing captured, and later passes run on mutated inputs with no diagnostic.
//
// hipMallocAsync is the interesting allocator here because it is what Kokkos selects by default on
// HIP < 7.0 and what PyTorch selects under expandable_segments, and on a VM-backed pool (the ROCm
// default) it reaches hsa_amd_vmem_map rather than hsa_amd_memory_pool_allocate.
TEST(kernel_replay_allocator_patterns, stream_ordered_allocation_is_snapshotted_or_counted)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    void* async_ptr = nullptr;
    if(hipMallocAsync(&async_ptr, kBufferBytes, stream) != hipSuccess || async_ptr == nullptr)
    {
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        GTEST_SKIP() << "hipMallocAsync unavailable on this runtime";
    }
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    const bool tracked   = in_snapshot_inventory(async_ptr, agent);
    const auto untracked = mt::untracked_device_memory(agent);
    const bool counted   = kernel_replay::memory_tracker::any_untracked(untracked);

    EXPECT_TRUE(tracked || counted)
        << "a stream-ordered allocation of " << kBufferBytes
        << " bytes is neither in the snapshot inventory nor counted as untracked, so a kernel "
           "writing to it would not be reverted between passes and nothing would report that";

    // If it went down the virtual-memory path it must not also be in the snapshot inventory:
    // snap/restore cannot reason about a range whose physical backing can be remapped underneath
    // it.
    if(untracked.vmem_regions > 0)
    {
        EXPECT_FALSE(tracked) << "a virtual-memory mapping must stay out of the snapshot inventory";
        EXPECT_GE(untracked.vmem_bytes, kBufferBytes)
            << "the counted virtual-memory footprint is smaller than the allocation";
    }

    EXPECT_EQ(hipFreeAsync(async_ptr, stream), hipSuccess);
    EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

// Same invariant for managed memory, which a kernel can write while it is resident on the device.
TEST(kernel_replay_allocator_patterns, managed_allocation_is_snapshotted_or_counted)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    float* managed = nullptr;
    if(hipMallocManaged(&managed, kBufferBytes) != hipSuccess || managed == nullptr)
        GTEST_SKIP() << "hipMallocManaged unavailable on this runtime";

    // Touch it from a kernel so it is genuinely device-resident rather than a host-side
    // reservation.
    kernel_launch::fill(managed, 3.0f, kElems);
    sync_ok();

    const bool tracked   = in_snapshot_inventory(managed, agent);
    const auto query     = mt::query_alloc(managed);
    const auto untracked = mt::untracked_device_memory(agent);

    // Managed memory backed by pinned host pages is out of scope rather than unsound: the agent
    // filter excludes it because it is not GPU-owned. Only assert the invariant for the case where
    // the runtime placed it on the device.
    if(!query.gpu_owned)
    {
        EXPECT_FALSE(tracked) << "host-resident managed memory must not enter a GPU agent snapshot";
        GTEST_SKIP() << "managed allocation is host-resident on this runtime (XNACK/pinned "
                        "placement), so the device-visible case is not exercised";
    }

    EXPECT_TRUE(tracked || kernel_replay::memory_tracker::any_untracked(untracked))
        << "a device-resident managed allocation is neither snapshottable nor counted as untracked";

    ASSERT_EQ(hipFree(managed), hipSuccess);
}

// Admission control has to be able to answer "what would this cost" without paying the cost. The
// estimate must match what snap() actually captures, or the budget check gates on the wrong number.
TEST(kernel_replay_allocator_patterns, footprint_estimate_matches_the_snapshot)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    float* buffer = nullptr;
    ASSERT_EQ(hipMalloc(&buffer, kBufferBytes), hipSuccess);
    kernel_launch::fill(buffer, 1.0f, kElems);
    sync_ok();

    const auto estimate = msnp::estimate_footprint(agent);
    const auto snapshot = msnp::snap(agent);
    ASSERT_TRUE(snapshot.ok);

    size_t captured_bytes = 0;
    for(const auto& block : snapshot.blocks)
        captured_bytes += block.host_copy.size();

    EXPECT_GE(estimate.bytes, kBufferBytes)
        << "the estimate omits the test's own allocation, so the budget check would let an "
           "oversized snapshot through";

    // Exact agreement is not required: the runtime allocates and frees between the two calls. The
    // estimate must be the right magnitude, or it is not usable for admission control.
    ASSERT_GT(captured_bytes, 0U);
    const double ratio = static_cast<double>(estimate.bytes) / static_cast<double>(captured_bytes);
    EXPECT_GT(ratio, 0.5) << "footprint estimate " << estimate.bytes
                          << " badly under-reports the captured " << captured_bytes << " bytes";
    EXPECT_LT(ratio, 2.0) << "footprint estimate " << estimate.bytes
                          << " badly over-reports the captured " << captured_bytes << " bytes";

    ASSERT_EQ(hipFree(buffer), hipSuccess);
}

// A kernel that writes memory outside the snapshot accumulates across passes. This test asserts the
// accumulation directly rather than asserting that it does not happen, because it *does* happen and
// the snapshot cannot prevent it -- which is exactly why the replay window declines instead.
// Pinning the behaviour here means a future change that makes such memory snapshottable will fail
// this test and force the decline policy to be revisited deliberately.
TEST(kernel_replay_allocator_patterns, untracked_memory_accumulates_across_passes)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    void* async_ptr = nullptr;
    if(hipMallocAsync(&async_ptr, kBufferBytes, stream) != hipSuccess || async_ptr == nullptr)
    {
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        GTEST_SKIP() << "hipMallocAsync unavailable on this runtime";
    }
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    auto* buffer = static_cast<float*>(async_ptr);
    kernel_launch::fill(buffer, 0.0f, kElems);
    sync_ok();

    if(in_snapshot_inventory(async_ptr, agent))
    {
        EXPECT_EQ(hipFreeAsync(async_ptr, stream), hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        GTEST_SKIP() << "this runtime's hipMallocAsync produced a snapshottable allocation, so the "
                        "untracked case is not exercised";
    }

    const auto snapshot = msnp::snap(agent);
    ASSERT_TRUE(snapshot.ok);

    // Three passes, each adding 1, with a restore between them exactly as the replay loop does.
    constexpr int kPasses = 3;
    for(int pass = 0; pass < kPasses; ++pass)
    {
        kernel_launch::add(buffer, 1.0f, kElems);
        sync_ok();
        ASSERT_TRUE(msnp::restore(snapshot));
    }

    const auto host = read_device(buffer, 1);
    EXPECT_FLOAT_EQ(host[0], static_cast<float>(kPasses))
        << "expected untracked memory to accumulate one increment per pass. If this now reads 0 "
           "the "
           "allocation became snapshottable, and the decline policy in the replay window should be "
           "reconsidered rather than this test relaxed";

    EXPECT_EQ(hipFreeAsync(async_ptr, stream), hipSuccess);
    EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}
