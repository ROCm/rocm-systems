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

#include "lib/common/container/lttng_record_header_buffer.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <typeinfo>
#include <vector>

namespace
{
using lttng_buffer_t = rocprofiler::common::container::lttng_record_header_buffer;

struct test_record
{
    uint64_t tid      = 0;
    uint64_t index    = 0;
    uint64_t checksum = 0;
};

test_record
make_record(uint64_t tid, uint64_t index)
{
    return test_record{tid, index, (tid << 32U) ^ (index * 1315423911ULL)};
}
}  // namespace

TEST(lttng_buffer, delegates_existing_buffer_abi)
{
    constexpr auto num_records = 128;
    auto           buffer      = lttng_buffer_t{num_records * (sizeof(test_record) + 128)};

    for(uint64_t i = 0; i < num_records; ++i)
    {
        auto record = make_record(3, i);
        ASSERT_TRUE(buffer.emplace(record));
    }

    EXPECT_EQ(buffer.get_num_record_headers(), num_records);

    auto seen = uint64_t{0};
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
    EXPECT_EQ(buffer.get_num_record_headers(), 0);
}

TEST(lttng_buffer, supports_parallel_producers)
{
    constexpr auto num_threads        = 8;
    constexpr auto records_per_thread = 128;
    constexpr auto num_records        = num_threads * records_per_thread;

    auto buffer = lttng_buffer_t{num_records * (sizeof(test_record) + 128)};
    auto ready  = std::atomic<int>{0};
    auto start  = std::atomic<bool>{false};
    auto threads = std::vector<std::thread>{};

    threads.reserve(num_threads);
    for(uint64_t tid = 0; tid < num_threads; ++tid)
    {
        threads.emplace_back([&, tid]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while(!start.load(std::memory_order_acquire))
            {}

            for(uint64_t i = 0; i < records_per_thread; ++i)
            {
                auto record = make_record(tid, i);
                ASSERT_TRUE(buffer.emplace(ROCPROFILER_BUFFER_CATEGORY_TRACING, tid, record));
            }
        });
    }

    while(ready.load(std::memory_order_acquire) != num_threads)
    {}
    start.store(true, std::memory_order_release);

    for(auto& itr : threads)
    {
        itr.join();
    }

    EXPECT_EQ(buffer.get_num_record_headers(), num_records);

    auto seen = std::atomic<uint64_t>{0};
    auto num_headers =
        buffer.process_record_headers(std::true_type{}, [&](auto&& records) {
            for(auto* header : records)
            {
                ASSERT_EQ(header->category, ROCPROFILER_BUFFER_CATEGORY_TRACING);
                auto* payload = static_cast<test_record*>(header->payload);
                ASSERT_NE(payload, nullptr);
                EXPECT_EQ(payload->checksum, make_record(payload->tid, payload->index).checksum);
                seen.fetch_add(1, std::memory_order_relaxed);
            }
        });

    EXPECT_EQ(num_headers, num_records);
    EXPECT_EQ(seen.load(), num_records);
}
