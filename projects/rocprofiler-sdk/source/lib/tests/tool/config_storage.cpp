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

#include "config_storage.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace
{
using rocprofiler::tool::config_details::immutable_config_storage;

struct test_config
{
    explicit test_config(uint64_t value = 0)
    : generation{value}
    {
        payload.fill(value);
    }

    uint64_t                 generation = 0;
    std::array<uint64_t, 32> payload    = {};
};

bool
is_consistent(const test_config& cfg)
{
    for(auto itr : cfg.payload)
        if(itr != cfg.generation) return false;
    return true;
}
}  // namespace

TEST(config_storage, published_generations_remain_valid)
{
    auto        storage = immutable_config_storage<test_config>{test_config{1}};
    const auto& first   = storage.get();

    const auto& second = storage.publish(test_config{2});

    EXPECT_EQ(first.generation, 1);
    EXPECT_TRUE(is_consistent(first));
    EXPECT_EQ(second.generation, 2);
    EXPECT_EQ(&storage.get(), &second);

    storage.reclaim_retired();

    EXPECT_EQ(storage.get().generation, 2);
    EXPECT_TRUE(is_consistent(storage.get()));
}

TEST(config_storage, readers_observe_complete_generations_during_publication)
{
    constexpr auto reader_count       = size_t{4};
    constexpr auto generation_count   = uint64_t{10000};
    auto           storage            = immutable_config_storage<test_config>{test_config{}};
    auto           start              = std::atomic<bool>{false};
    auto           done               = std::atomic<bool>{false};
    auto           inconsistent_reads = std::atomic<size_t>{0};
    auto           readers            = std::vector<std::thread>{};

    readers.reserve(reader_count);
    for(size_t i = 0; i < reader_count; ++i)
    {
        readers.emplace_back([&]() {
            while(!start.load(std::memory_order_acquire))
                std::this_thread::yield();

            while(!done.load(std::memory_order_acquire))
            {
                if(!is_consistent(storage.get()))
                    inconsistent_reads.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    start.store(true, std::memory_order_release);
    for(uint64_t generation = 1; generation <= generation_count; ++generation)
        storage.publish(test_config{generation});
    done.store(true, std::memory_order_release);

    for(auto& itr : readers)
        itr.join();

    EXPECT_EQ(inconsistent_reads.load(), 0);
    EXPECT_EQ(storage.get().generation, generation_count);
    EXPECT_TRUE(is_consistent(storage.get()));
}
