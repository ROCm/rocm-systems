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

/**
 * @file buffer_correlation_ordering.cpp
 * @brief Tests for correlation ID ordering: API record must be seen before retirement
 *
 * This test reproduces the "internal correlation id was retired prematurely" error.
 *
 * The bug: With an array-based double buffer design, the buffer index selection
 * and write are not atomic. A retirement record can land in a "newer" buffer
 * than the API record. If the newer buffer's callback runs first, the client
 * sees the retirement before the API record.
 */

#include "lib/common/units.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <set>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace
{
// Record type markers
constexpr uint32_t RECORD_TYPE_API        = 1;
constexpr uint32_t RECORD_TYPE_RETIREMENT = 2;

// API record structure
struct alignas(8) api_record_t
{
    uint64_t correlation_id;
    uint64_t thread_id;
    uint64_t timestamp;
    uint64_t padding[5];
};

static_assert(sizeof(api_record_t) == 64, "api_record_t should be 64 bytes");

// Retirement record structure
struct alignas(8) retirement_record_t
{
    uint64_t correlation_id;
    uint64_t thread_id;
    uint64_t timestamp;
    uint64_t padding[5];
};

static_assert(sizeof(retirement_record_t) == 64, "retirement_record_t should be 64 bytes");

// Callback data
struct callback_data_t
{
    std::mutex                             mutex;
    std::set<uint64_t>                     seen_api_ids;
    std::atomic<uint64_t>                  ordering_violations{0};
    std::atomic<uint64_t>                  total_api_records{0};
    std::atomic<uint64_t>                  total_retirement_records{0};
    std::atomic<uint64_t>                  callback_count{0};
    std::vector<std::pair<uint64_t, bool>> violation_details;
};

// Buffer callback
void
test_buffer_callback(rocprofiler_context_id_t,
                     rocprofiler_buffer_id_t,
                     rocprofiler_record_header_t** headers,
                     size_t                        num_headers,
                     void*                         user_data,
                     uint64_t)
{
    auto* data = static_cast<callback_data_t*>(user_data);
    data->callback_count.fetch_add(1, std::memory_order_relaxed);

    // Simulate slow flush to increase race window
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    auto lock = std::lock_guard<std::mutex>{data->mutex};

    for(size_t i = 0; i < num_headers; ++i)
    {
        if(!headers[i] || !headers[i]->payload) continue;

        if(headers[i]->category == RECORD_TYPE_API)
        {
            auto* record = static_cast<const api_record_t*>(headers[i]->payload);
            data->seen_api_ids.insert(record->correlation_id);
            data->total_api_records.fetch_add(1, std::memory_order_relaxed);
        }
        else if(headers[i]->category == RECORD_TYPE_RETIREMENT)
        {
            auto* record = static_cast<const retirement_record_t*>(headers[i]->payload);

            if(data->seen_api_ids.find(record->correlation_id) == data->seen_api_ids.end())
            {
                data->ordering_violations.fetch_add(1, std::memory_order_relaxed);
                data->violation_details.emplace_back(record->correlation_id, true);

                std::cerr << "ORDERING VIOLATION: Correlation ID " << record->correlation_id
                          << " was retired prematurely" << std::endl;
            }
            data->total_retirement_records.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

uint64_t
get_timestamp_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Detect if using deque-based pool design (has pool_size member)
template <typename T, typename = void>
struct has_pool_size : std::false_type
{};

template <typename T>
struct has_pool_size<T, std::void_t<decltype(std::declval<T>().pool_size)>> : std::true_type
{};

// Initialize buffers - works with both array and deque designs
template <typename BufferInstance>
void
initialize_buffers(BufferInstance* buffer_v, size_t buffer_size)
{
    if constexpr(has_pool_size<BufferInstance>::value)
    {
        // Deque-based pool design
        buffer_v->buffer_size = buffer_size;
        buffer_v->pool_size   = 2;
        for(size_t i = 0; i < buffer_v->pool_size; ++i)
        {
            buffer_v->buffers.emplace_back();
            buffer_v->buffers.back().allocate(buffer_size);
        }
    }
    else
    {
        // Array-based design
        for(auto& buf : buffer_v->buffers)
        {
            buf.allocate(buffer_size);
        }
    }
}

}  // namespace

/**
 * @test CorrelationOrderingTest
 * @brief Test that retirement records are never seen before their corresponding API records.
 */
TEST(buffer_correlation_ordering, api_before_retirement)
{
    namespace buffer = ::rocprofiler::buffer;
    namespace common = ::rocprofiler::common;

    constexpr size_t num_threads           = 4;
    constexpr size_t operations_per_thread = 200;
    constexpr size_t buffer_size           = 256;
    constexpr size_t watermark             = 2;

    auto buffer_id = buffer::allocate_buffer();
    ASSERT_TRUE(buffer_id) << "Failed to allocate buffer";

    auto* buffer_v = buffer::get_buffer(*buffer_id);
    ASSERT_NE(buffer_v, nullptr);

    buffer_v->watermark = watermark;
    buffer_v->policy    = ROCPROFILER_BUFFER_POLICY_LOSSLESS;
    initialize_buffers(buffer_v, buffer_size);

    callback_data_t callback_data;
    buffer_v->callback      = test_buffer_callback;
    buffer_v->callback_data = &callback_data;

    std::atomic<uint64_t> next_correlation_id{1};

    auto worker_func = [&](size_t thread_idx) {
        for(size_t i = 0; i < operations_per_thread; ++i)
        {
            uint64_t corr_id = next_correlation_id.fetch_add(1, std::memory_order_acq_rel);

            api_record_t api_record;
            api_record.correlation_id = corr_id;
            api_record.thread_id      = thread_idx;
            api_record.timestamp      = get_timestamp_ns();

            bool success1 = buffer_v->emplace(RECORD_TYPE_API, 1, api_record);
            EXPECT_TRUE(success1) << "Failed to emplace API record";

            if(i % 5 == 0)
            {
                std::this_thread::yield();
            }

            retirement_record_t retirement_record;
            retirement_record.correlation_id = corr_id;
            retirement_record.thread_id      = thread_idx;
            retirement_record.timestamp      = get_timestamp_ns();

            bool success2 = buffer_v->emplace(RECORD_TYPE_RETIREMENT, 1, retirement_record);
            EXPECT_TRUE(success2) << "Failed to emplace retirement record";
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(worker_func, t);
    }

    for(auto& t : threads)
    {
        t.join();
    }

    auto flush_status = buffer::flush(*buffer_id, true);
    EXPECT_EQ(flush_status, ROCPROFILER_STATUS_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    const size_t expected_ops = num_threads * operations_per_thread;
    std::cerr << "\n=== Correlation Ordering Test Results ===" << std::endl;
    std::cerr << "Threads: " << num_threads << std::endl;
    std::cerr << "Operations per thread: " << operations_per_thread << std::endl;
    std::cerr << "Buffer size: " << buffer_size << " bytes" << std::endl;
    std::cerr << "Watermark: " << watermark << " records" << std::endl;
    std::cerr << "Callback invocations: " << callback_data.callback_count.load() << std::endl;
    std::cerr << "Total API records: " << callback_data.total_api_records.load() << std::endl;
    std::cerr << "Total retirement records: " << callback_data.total_retirement_records.load()
              << std::endl;
    std::cerr << "ORDERING VIOLATIONS: " << callback_data.ordering_violations.load() << std::endl;

    EXPECT_EQ(callback_data.total_api_records.load(), expected_ops) << "Missing API records";
    EXPECT_EQ(callback_data.total_retirement_records.load(), expected_ops)
        << "Missing retirement records";

    EXPECT_EQ(callback_data.ordering_violations.load(), 0)
        << "FAILED: Found " << callback_data.ordering_violations.load()
        << " correlation ID ordering violations!";

    auto destroy_status = rocprofiler_destroy_buffer(*buffer_id);
    EXPECT_EQ(destroy_status, ROCPROFILER_STATUS_SUCCESS);
}

/**
 * @test HighContentionCorrelationTest
 * @brief Stress test with maximum contention.
 */
TEST(buffer_correlation_ordering, high_contention)
{
    namespace buffer = ::rocprofiler::buffer;
    namespace common = ::rocprofiler::common;

    constexpr size_t num_threads           = 8;
    constexpr size_t operations_per_thread = 100;
    constexpr size_t buffer_size           = 128;
    constexpr size_t watermark             = 1;

    auto buffer_id = buffer::allocate_buffer();
    ASSERT_TRUE(buffer_id);

    auto* buffer_v = buffer::get_buffer(*buffer_id);
    ASSERT_NE(buffer_v, nullptr);

    buffer_v->watermark = watermark;
    buffer_v->policy    = ROCPROFILER_BUFFER_POLICY_LOSSLESS;
    initialize_buffers(buffer_v, buffer_size);

    callback_data_t callback_data;
    buffer_v->callback      = test_buffer_callback;
    buffer_v->callback_data = &callback_data;

    std::atomic<uint64_t> next_correlation_id{1};

    auto worker_func = [&](size_t thread_idx) {
        for(size_t i = 0; i < operations_per_thread; ++i)
        {
            uint64_t corr_id = next_correlation_id.fetch_add(1, std::memory_order_acq_rel);

            api_record_t api_record;
            api_record.correlation_id = corr_id;
            api_record.thread_id      = thread_idx;
            api_record.timestamp      = get_timestamp_ns();
            buffer_v->emplace(RECORD_TYPE_API, 1, api_record);

            retirement_record_t retirement_record;
            retirement_record.correlation_id = corr_id;
            retirement_record.thread_id      = thread_idx;
            retirement_record.timestamp      = get_timestamp_ns();
            buffer_v->emplace(RECORD_TYPE_RETIREMENT, 1, retirement_record);
        }
    };

    std::vector<std::thread> threads;
    for(size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(worker_func, t);
    }

    for(auto& t : threads)
    {
        t.join();
    }

    buffer::flush(*buffer_id, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    const size_t expected_ops = num_threads * operations_per_thread;
    std::cerr << "\n=== High Contention Test Results ===" << std::endl;
    std::cerr << "Callback invocations: " << callback_data.callback_count.load() << std::endl;
    std::cerr << "ORDERING VIOLATIONS: " << callback_data.ordering_violations.load() << std::endl;

    EXPECT_EQ(callback_data.total_api_records.load(), expected_ops);
    EXPECT_EQ(callback_data.total_retirement_records.load(), expected_ops);

    EXPECT_EQ(callback_data.ordering_violations.load(), 0)
        << "FAILED: Found ordering violations under high contention!";

    rocprofiler_destroy_buffer(*buffer_id);
}

/**
 * @test SingleThreadOrderingTest
 * @brief Single-threaded test for basic ordering correctness.
 */
TEST(buffer_correlation_ordering, single_thread_ordering)
{
    namespace buffer = ::rocprofiler::buffer;
    namespace common = ::rocprofiler::common;

    constexpr size_t num_operations = 500;
    constexpr size_t buffer_size    = 128;
    constexpr size_t watermark      = 1;

    auto buffer_id = buffer::allocate_buffer();
    ASSERT_TRUE(buffer_id);

    auto* buffer_v = buffer::get_buffer(*buffer_id);
    ASSERT_NE(buffer_v, nullptr);

    buffer_v->watermark = watermark;
    buffer_v->policy    = ROCPROFILER_BUFFER_POLICY_LOSSLESS;
    initialize_buffers(buffer_v, buffer_size);

    callback_data_t callback_data;
    buffer_v->callback      = test_buffer_callback;
    buffer_v->callback_data = &callback_data;

    for(size_t i = 0; i < num_operations; ++i)
    {
        uint64_t corr_id = i + 1;

        api_record_t api_record;
        api_record.correlation_id = corr_id;
        api_record.thread_id      = 0;
        api_record.timestamp      = get_timestamp_ns();
        buffer_v->emplace(RECORD_TYPE_API, 1, api_record);

        retirement_record_t retirement_record;
        retirement_record.correlation_id = corr_id;
        retirement_record.thread_id      = 0;
        retirement_record.timestamp      = get_timestamp_ns();
        buffer_v->emplace(RECORD_TYPE_RETIREMENT, 1, retirement_record);
    }

    buffer::flush(*buffer_id, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    std::cerr << "\n=== Single Thread Test Results ===" << std::endl;
    std::cerr << "Callback invocations: " << callback_data.callback_count.load() << std::endl;
    std::cerr << "API records: " << callback_data.total_api_records.load() << std::endl;
    std::cerr << "Retirement records: " << callback_data.total_retirement_records.load()
              << std::endl;
    std::cerr << "ORDERING VIOLATIONS: " << callback_data.ordering_violations.load() << std::endl;

    EXPECT_EQ(callback_data.total_api_records.load(), num_operations);
    EXPECT_EQ(callback_data.total_retirement_records.load(), num_operations);
    EXPECT_EQ(callback_data.ordering_violations.load(), 0)
        << "FAILED: Found ordering violations in single-threaded test!";

    rocprofiler_destroy_buffer(*buffer_id);
}
