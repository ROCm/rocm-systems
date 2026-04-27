// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/container/user_ring_record_header_buffer.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <thread>
#include <vector>

namespace
{
using user_ring_buffer_t = rocprofiler::common::container::user_ring_record_header_buffer;

struct test_record
{
    uint64_t tid      = 0;
    uint64_t index    = 0;
    uint64_t checksum = 0;
    uint64_t pad[5]   = {};
};

test_record
make_record(uint64_t tid, uint64_t index)
{
    auto ret     = test_record{};
    ret.tid      = tid;
    ret.index    = index;
    ret.checksum = (tid << 32) ^ index ^ 0x9e3779b97f4a7c15ULL;
    for(size_t i = 0; i < std::size(ret.pad); ++i)
        ret.pad[i] = ret.checksum + i;
    return ret;
}
}  // namespace

TEST(user_ring_buffer, serial)
{
    constexpr auto num_records = 1024;
    auto           buffer      = user_ring_buffer_t{num_records * (sizeof(test_record) + 128)};

    for(uint64_t i = 0; i < num_records; ++i)
    {
        auto record = make_record(0, i);
        ASSERT_TRUE(buffer.emplace(record));
    }

    EXPECT_EQ(buffer.get_num_record_headers(), num_records);

    auto seen = size_t{0};
    auto num_headers =
        buffer.process_record_headers(std::true_type{}, [&](auto&& records) {
            for(auto* header : records)
            {
                ASSERT_EQ(header->hash, typeid(test_record).hash_code());
                auto* payload = static_cast<test_record*>(header->payload);
                ASSERT_NE(payload, nullptr);
                EXPECT_EQ(payload->checksum, make_record(payload->tid, payload->index).checksum);
                ++seen;
            }
        });

    EXPECT_EQ(num_headers, num_records);
    EXPECT_EQ(seen, num_records);
    EXPECT_TRUE(buffer.is_empty());
}

TEST(user_ring_buffer, parallel_per_thread_shards)
{
    constexpr auto num_threads        = 16;
    constexpr auto records_per_thread = 256;
    constexpr auto num_records        = num_threads * records_per_thread;

    auto buffer = user_ring_buffer_t{num_records * (sizeof(test_record) + 128)};
    auto ready  = std::atomic<uint32_t>{0};
    auto go     = std::atomic<bool>{false};
    auto thrs   = std::vector<std::thread>{};
    thrs.reserve(num_threads);

    for(uint64_t tid = 0; tid < num_threads; ++tid)
    {
        thrs.emplace_back([&, tid]() {
            ready.fetch_add(1, std::memory_order_release);
            while(!go.load(std::memory_order_acquire)) std::this_thread::yield();

            for(uint64_t i = 0; i < records_per_thread; ++i)
            {
                auto record = make_record(tid, i);
                ASSERT_TRUE(buffer.emplace(1, 1, record));
            }
        });
    }

    while(ready.load(std::memory_order_acquire) != num_threads) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    for(auto& itr : thrs)
        itr.join();

    EXPECT_EQ(buffer.get_num_record_headers(), num_records);

    auto seen = std::atomic<size_t>{0};
    auto num_headers =
        buffer.process_record_headers(std::true_type{}, [&](auto&& records) {
            for(auto* header : records)
            {
                ASSERT_EQ(header->category, 1);
                ASSERT_EQ(header->kind, 1);
                auto* payload = static_cast<test_record*>(header->payload);
                ASSERT_NE(payload, nullptr);
                EXPECT_EQ(payload->checksum, make_record(payload->tid, payload->index).checksum);
                seen.fetch_add(1, std::memory_order_relaxed);
            }
        });

    EXPECT_EQ(num_headers, num_records);
    EXPECT_EQ(seen.load(), num_records);
}

TEST(user_ring_buffer, save_load)
{
    constexpr auto num_records = 64;
    auto           buffer      = user_ring_buffer_t{num_records * (sizeof(test_record) + 128)};

    for(uint64_t i = 0; i < num_records; ++i)
    {
        auto record = make_record(7, i);
        ASSERT_TRUE(buffer.emplace(record));
    }

    {
        auto ofs =
            std::fstream{"user-ring-buffer-save-load.dat", std::ios::out | std::ios::binary};
        buffer.save(ofs);
    }

    ASSERT_EQ(buffer.clear(), num_records);
    EXPECT_EQ(buffer.get_num_record_headers(), 0);

    {
        auto ifs =
            std::fstream{"user-ring-buffer-save-load.dat", std::ios::in | std::ios::binary};
        buffer.load(ifs);
    }

    EXPECT_EQ(buffer.get_num_record_headers(), num_records);

    auto num_headers =
        buffer.process_record_headers(std::false_type{}, [&](auto&& records) {
            for(auto* header : records)
            {
                ASSERT_EQ(header->hash, typeid(test_record).hash_code());
                auto* payload = static_cast<test_record*>(header->payload);
                ASSERT_NE(payload, nullptr);
                EXPECT_EQ(payload->checksum, make_record(payload->tid, payload->index).checksum);
            }
        });

    EXPECT_EQ(num_headers, num_records);
}
