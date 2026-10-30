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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/pending_batches.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

namespace
{
using rocprofiler::hsa::pending_batches;
using namespace std::chrono_literals;
}  // namespace

TEST(pending_batches, empty_wait_returns_immediately)
{
    auto batches = pending_batches{};
    EXPECT_TRUE(batches.wait_for_added(1h, 1h));
}

// A later batch finishing first must not satisfy a wait for an earlier one.
TEST(pending_batches, out_of_order_completion_does_not_end_wait)
{
    auto batches = pending_batches{};
    auto first   = batches.add();
    auto second  = batches.add();

    batches.set_emitting(second);
    batches.remove(second);

    const auto t0 = pending_batches::clock_type::now();
    EXPECT_FALSE(batches.wait_for_added(20ms, 1h));
    EXPECT_GE(pending_batches::clock_type::now() - t0, 20ms);

    batches.set_emitting(first);
    batches.remove(first);
    EXPECT_TRUE(batches.wait_for_added(1h, 1h));
    EXPECT_TRUE(batches.empty());
}

// Batches added after the wait started are not waited for.
TEST(pending_batches, ignores_batches_added_during_wait)
{
    auto batches = pending_batches{};
    auto first   = batches.add();

    auto worker = std::thread{[&batches, first]() {
        std::this_thread::sleep_for(5ms);
        auto later = batches.add();
        batches.set_emitting(first);
        batches.remove(first);
        (void) later;
    }};

    EXPECT_TRUE(batches.wait_for_added(1h, 1h));
    worker.join();
    EXPECT_FALSE(batches.empty());
}

// Progress resets the timeout: a long emission made of many short steps completes.
TEST(pending_batches, progress_extends_wait)
{
    auto batches = pending_batches{};
    auto ids     = std::vector<uint64_t>{};
    for(int i = 0; i < 20; ++i)
        ids.emplace_back(batches.add());

    auto worker = std::thread{[&batches, &ids]() {
        for(auto id : ids)
        {
            std::this_thread::sleep_for(2ms);
            batches.set_emitting(id);
            batches.remove(id);
        }
    }};

    EXPECT_TRUE(batches.wait_for_added(10ms, 10ms));
    worker.join();
}

// A batch whose records are being emitted gets the emit timeout, not the device one.
TEST(pending_batches, emitting_uses_emit_timeout)
{
    auto batches = pending_batches{};
    auto id      = batches.add();
    batches.set_emitting(id);

    auto worker = std::thread{[&batches, id]() {
        std::this_thread::sleep_for(30ms);
        batches.remove(id);
    }};

    EXPECT_TRUE(batches.wait_for_added(1ms, 1h));
    worker.join();
}

// Id 0 marks an untracked batch: it is never handed out, and transitions on it are not progress.
TEST(pending_batches, untracked_id_is_ignored)
{
    auto batches = pending_batches{};
    auto id      = batches.add();
    EXPECT_NE(id, 0);

    batches.set_emitting(0);
    batches.remove(0);
    EXPECT_FALSE(batches.wait_for_added(5ms, 5ms));

    batches.remove(id);
    EXPECT_TRUE(batches.empty());
}
