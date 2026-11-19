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

// Unit tests for the kernel-replay snapshot staging pool. The pool is host memory management only,
// so these run without a GPU. Each test uses its own fake agent handle so the process-wide pool
// state of one test cannot leak into another.

#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <utility>

namespace msnp = rocprofiler::kernel_replay::memory_snapshot;

namespace
{
constexpr size_t kib = 1024;
constexpr size_t mib = 1024 * kib;
}  // namespace

// A fresh buffer is sized to the request and holds writable storage.
TEST(kernel_replay_staging, acquire_sizes_storage_to_the_request)
{
    const auto agent = hsa_agent_t{.handle = 0x5701};

    auto buffer = msnp::staging_buffer_t{};
    ASSERT_TRUE(buffer.acquire(agent, 3 * mib));
    EXPECT_EQ(buffer.size(), 3 * mib);
    ASSERT_NE(buffer.data(), nullptr);
    std::memset(buffer.data(), 0x5a, buffer.size());
    EXPECT_EQ(msnp::retained_staging_bytes(agent), 0u);
}

// Destroying a buffer hands its storage to the agent's pool, and the next snapshot of the same
// footprint gets the very same storage back instead of allocating again.
TEST(kernel_replay_staging, destroyed_storage_is_reused_by_the_next_acquire)
{
    const auto agent = hsa_agent_t{.handle = 0x5702};

    const char* first = nullptr;
    {
        auto buffer = msnp::staging_buffer_t{};
        ASSERT_TRUE(buffer.acquire(agent, 8 * mib));
        first = buffer.data();
    }
    EXPECT_EQ(msnp::retained_staging_bytes(agent), 8 * mib);

    auto again = msnp::staging_buffer_t{};
    ASSERT_TRUE(again.acquire(agent, 8 * mib));
    EXPECT_EQ(again.data(), first);
    EXPECT_EQ(msnp::retained_staging_bytes(agent), 0u);
}

// Reuse is best fit, capped at twice the request: a small region must not take a large buffer that
// a later, larger region would then have to allocate again.
TEST(kernel_replay_staging, reuse_is_best_fit_within_twice_the_request)
{
    const auto agent = hsa_agent_t{.handle = 0x5703};

    const char* large = nullptr;
    {
        auto buffer = msnp::staging_buffer_t{};
        ASSERT_TRUE(buffer.acquire(agent, 100 * kib));
        large = buffer.data();
    }

    auto too_small = msnp::staging_buffer_t{};
    ASSERT_TRUE(too_small.acquire(agent, 40 * kib));
    EXPECT_NE(too_small.data(), large);
    EXPECT_EQ(msnp::retained_staging_bytes(agent), 100 * kib);

    auto fits = msnp::staging_buffer_t{};
    ASSERT_TRUE(fits.acquire(agent, 60 * kib));
    EXPECT_EQ(fits.data(), large);
    EXPECT_EQ(fits.size(), 60 * kib);
    EXPECT_EQ(msnp::retained_staging_bytes(agent), 0u);
}

// Pools are per agent: storage retained for one GPU is never handed to another.
TEST(kernel_replay_staging, pools_are_per_agent)
{
    const auto agent = hsa_agent_t{.handle = 0x5704};
    const auto other = hsa_agent_t{.handle = 0x5705};

    const char* retained = nullptr;
    {
        auto buffer = msnp::staging_buffer_t{};
        ASSERT_TRUE(buffer.acquire(agent, 2 * mib));
        retained = buffer.data();
    }

    auto elsewhere = msnp::staging_buffer_t{};
    ASSERT_TRUE(elsewhere.acquire(other, 2 * mib));
    EXPECT_NE(elsewhere.data(), retained);
    EXPECT_EQ(msnp::retained_staging_bytes(agent), 2 * mib);
}

// Moving a buffer transfers its storage; only the final owner returns it to the pool, once.
TEST(kernel_replay_staging, moves_return_storage_exactly_once)
{
    const auto agent = hsa_agent_t{.handle = 0x5706};

    {
        auto source = msnp::staging_buffer_t{};
        ASSERT_TRUE(source.acquire(agent, 4 * mib));
        const char* storage = source.data();

        auto moved = msnp::staging_buffer_t{std::move(source)};
        EXPECT_EQ(moved.data(), storage);
        EXPECT_EQ(moved.size(), 4 * mib);
        EXPECT_EQ(source.size(), 0u);  // NOLINT(bugprone-use-after-move)

        auto assigned = msnp::staging_buffer_t{};
        ASSERT_TRUE(assigned.acquire(agent, 1 * mib));
        assigned = std::move(moved);
        EXPECT_EQ(assigned.data(), storage);
        // The 1 MiB storage `assigned` held before the move went back to the pool.
        EXPECT_EQ(msnp::retained_staging_bytes(agent), 1 * mib);
    }
    EXPECT_EQ(msnp::retained_staging_bytes(agent), 5 * mib);
}

// snap() frees whatever retained staging it did not take, even when it declines the snapshot, so
// host memory held between replayed dispatches never exceeds the last snapshot's footprint. Without
// a live HSA runtime the module-variable scan cannot run and the snapshot is declined.
TEST(kernel_replay_staging, snap_frees_retained_storage_it_does_not_reuse)
{
    const auto agent = hsa_agent_t{.handle = 0x5707};

    {
        auto buffer = msnp::staging_buffer_t{};
        ASSERT_TRUE(buffer.acquire(agent, 16 * mib));
    }
    ASSERT_EQ(msnp::retained_staging_bytes(agent), 16 * mib);

    const auto snapshot = msnp::snap(agent);
    EXPECT_FALSE(snapshot.ok);
    EXPECT_EQ(msnp::retained_staging_bytes(agent), 0u);
}
