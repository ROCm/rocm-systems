// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <mutex>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/core.h>
#include <gtest/gtest.h>

#include "lib/rocprofiler-sdk/counters/sample_consumer.hpp"

namespace rocprofiler
{
namespace counters
{
constexpr size_t NUM_THREADS  = 5;
constexpr size_t NUM_ELEMENTS = 1ul << 17;
using result_array_t          = std::array<std::atomic<size_t>, NUM_ELEMENTS>;
using result_array_ptr_t      = std::shared_ptr<result_array_t>;

struct DummyData
{
    size_t             index;
    size_t             increment;
    result_array_ptr_t array;
};

using consumer_t = consumer_thread_t<DummyData>;

void
consume_fn(DummyData&& data)
{
    data.array->at(data.index).fetch_add(data.increment);
}

TEST(consumer, nothread)
{
    auto array = std::make_shared<result_array_t>();

    consumer_t consumer(consume_fn);
    consumer.add(DummyData{1, 1, array});

    EXPECT_EQ(array->at(0).load(), 0);
    EXPECT_EQ(array->at(1).load(), 1);
}

TEST(consumer, singlethread)
{
    auto array = std::make_shared<result_array_t>();

    {
        consumer_t consumer(consume_fn);
        consumer.start();

        for(size_t i = 0; i < NUM_ELEMENTS; i++)
            consumer.add(DummyData{i, 1, array});
    }

    for(auto& var : *array)
        EXPECT_EQ(var.load(), 1);
}

TEST(consumer, multithreaded)
{
    auto       array = std::make_shared<result_array_t>();
    consumer_t consumer(consume_fn);

    auto produce_fn = [&](size_t tid) {
        for(size_t i = 0; i < NUM_ELEMENTS; i++)
            consumer.add(DummyData{i, tid, array});
    };

    {
        std::vector<std::future<void>> threads{};
        for(size_t i = 0; i < NUM_THREADS; i++)
            threads.push_back(std::async(std::launch::async, produce_fn, i + 1));

        consumer.start();
    }

    consumer.exit();

    size_t expected = NUM_THREADS * (NUM_THREADS + 1) / 2;

    for(auto& var : *array)
        EXPECT_EQ(var.load(), expected);
}

// Verifies that a single consumer instance survives multiple
// start()/exit() cycles. Each cycle adds 1 to every slot, so after
// CYCLES iterations every slot must equal CYCLES. Exercises the
// drain-on-exit logic (the `exited` flag + cv wait) followed by a
// fresh start() that re-arms it.
TEST(consumer, restart)
{
    auto array = std::make_shared<result_array_t>();

    consumer_t    consumer(consume_fn);
    constexpr int CYCLES = 3;

    for(int cycle = 0; cycle < CYCLES; ++cycle)
    {
        consumer.start();
        for(size_t i = 0; i < NUM_ELEMENTS; ++i)
            consumer.add(DummyData{i, 1, array});
        consumer.exit();
    }

    for(auto& var : *array)
        EXPECT_EQ(var.load(), static_cast<size_t>(CYCLES));
}

// Verifies that calling add() after exit() does not lose work: the
// item must be processed synchronously on the calling thread via the
// fallback path (`!valid` clause in add()).
TEST(consumer, add_after_exit)
{
    auto array = std::make_shared<result_array_t>();

    consumer_t consumer(consume_fn);
    consumer.start();
    consumer.exit();

    consumer.add(DummyData{42, 7, array});

    EXPECT_EQ(array->at(42).load(), 7u);
    EXPECT_EQ(array->at(0).load(), 0u);
}

// Verifies the synchronous fallback when the ring buffer overflows.
// Throttles the consumer so the producer outpaces it; asserts that
// (a) every item is processed exactly once and (b) at least some
// items took the fallback path (consumed on the producer thread).
TEST(consumer, buffer_full_fallback)
{
    auto                array = std::make_shared<result_array_t>();
    std::atomic<size_t> fallback_count{0};
    std::atomic<size_t> processed{0};
    constexpr size_t    SLOW_FIRST_N = 256;

    const auto producer_tid = std::this_thread::get_id();

    auto slow_consume = [&, producer_tid](DummyData&& data) {
        if(std::this_thread::get_id() == producer_tid) fallback_count.fetch_add(1);

        if(processed.fetch_add(1) < SLOW_FIRST_N)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        data.array->at(data.index).fetch_add(data.increment);
    };

    consumer_thread_t<DummyData> consumer(slow_consume);
    consumer.start();

    for(size_t i = 0; i < NUM_ELEMENTS; ++i)
        consumer.add(DummyData{i, 1, array});

    consumer.exit();

    for(auto& var : *array)
        EXPECT_EQ(var.load(), 1u);

    EXPECT_GT(fallback_count.load(), 0u);
}

}  // namespace counters
}  // namespace rocprofiler
