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
 * @file buffer_ordering_stress.cpp
 * @brief Stress tests for buffer flush ordering
 *
 * These tests expose the flush ordering bug in the original double-buffer implementation
 * which uses a simple atomic index that wraps around without proper synchronization.
 *
 * The bug: When buffer 0 is full and flushing, writes go to buffer 1. If buffer 1
 * fills up before buffer 0's flush completes, the index wraps back to buffer 0,
 * which may still be flushing. Independent flush tasks submitted to the thread pool
 * can complete out of order, causing:
 * 1. FIFO ordering violation - newer records may be processed before older ones
 * 2. Per-thread sequence ordering violations in the callback
 *
 * The fix uses a mutex to serialize flush submissions and ensures all pending flushes
 * complete before submitting new ones, guaranteeing FIFO callback ordering.
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
#include <deque>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <type_traits>
#include <vector>

namespace
{
// Detect if we're using the new deque-based pool design or the old array-based design
template <typename T, typename = void>
struct has_pool_size : std::false_type
{};

template <typename T>
struct has_pool_size<T, std::void_t<decltype(std::declval<T>().pool_size)>> : std::true_type
{};

// Test record structure with sequence numbers for ordering verification
struct alignas(8) test_record_t
{
    uint64_t thread_id;     // Thread that created this record
    uint64_t sequence_num;  // Per-thread sequence number
    uint64_t global_seq;    // Global sequence number (when written)
    uint64_t timestamp;     // Timestamp when record was created
    uint64_t padding[4];    // Padding to make record larger (fills buffers faster)
};

static_assert(sizeof(test_record_t) == 64, "test_record_t should be 64 bytes");

// Callback data structure to collect all received records
struct callback_data_t
{
    std::mutex                            mutex;
    std::vector<test_record_t>            records;
    std::atomic<uint64_t>                 callback_count{0};
    std::atomic<uint64_t>                 total_records{0};
    std::chrono::steady_clock::time_point start_time;
};

// Buffer callback that collects records
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

    // Simulate slow flush processing to increase race window
    // This makes the ordering bug more likely to manifest
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    auto lock = std::lock_guard<std::mutex>{data->mutex};
    for(size_t i = 0; i < num_headers; ++i)
    {
        if(headers[i] && headers[i]->payload)
        {
            auto* record = static_cast<const test_record_t*>(headers[i]->payload);
            data->records.push_back(*record);
            data->total_records.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// Get current timestamp in nanoseconds
uint64_t
get_timestamp_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Helper to initialize buffers - works with both array (old) and deque (new) designs
template <typename BufferInstance>
void
initialize_buffers(BufferInstance* buffer_v, size_t buffer_size)
{
    // Check if this is the new pool design (has pool_size member and deque)
    // or old array design (fixed array size)
    if constexpr(has_pool_size<BufferInstance>::value)
    {
        // New deque-based pool design
        buffer_v->pool_size = 2;
        for(size_t i = 0; i < buffer_v->pool_size; ++i)
        {
            buffer_v->buffers.emplace_back();
            buffer_v->buffers.back().allocate(buffer_size);
        }
    }
    else
    {
        // Old array-based design (size is fixed at compile time)
        for(auto& buf : buffer_v->buffers)
        {
            buf.allocate(buffer_size);
        }
    }
}

}  // namespace

/**
 * @test MultiThreadOrderingTest
 * @brief Test that records from multiple threads are received in correct per-thread order.
 *
 * This test exposes the FIFO ordering bug in origin/develop:
 * - Multiple threads write records with per-thread sequence numbers
 * - Each thread's records should be received in increasing sequence order
 * - The old implementation may flush buffers out of order (buffer 1 before buffer 0)
 *   causing per-thread ordering violations
 */
TEST(buffer_ordering_stress, multi_thread_ordering)
{
    namespace buffer = ::rocprofiler::buffer;
    namespace common = ::rocprofiler::common;

    constexpr size_t num_threads        = 4;
    constexpr size_t records_per_thread = 500;
    constexpr size_t buffer_size        = 1024;  // Small buffer to force frequent flushes
    constexpr size_t watermark          = 4;     // Very low watermark = frequent rotations

    // Allocate buffer
    auto buffer_id = buffer::allocate_buffer();
    ASSERT_TRUE(buffer_id) << "Failed to allocate buffer";

    auto* buffer_v = buffer::get_buffer(*buffer_id);
    ASSERT_NE(buffer_v, nullptr);

    // Initialize buffer
    buffer_v->watermark = watermark;
    buffer_v->policy    = ROCPROFILER_BUFFER_POLICY_LOSSLESS;
    initialize_buffers(buffer_v, buffer_size);

    // Set up callback
    callback_data_t callback_data;
    callback_data.start_time = std::chrono::steady_clock::now();
    buffer_v->callback       = test_buffer_callback;
    buffer_v->callback_data  = &callback_data;

    // Global sequence counter for tracking write order
    std::atomic<uint64_t> global_seq{0};

    // Thread function: write records with increasing sequence numbers
    auto writer_func = [&](size_t thread_idx) {
        for(size_t i = 0; i < records_per_thread; ++i)
        {
            test_record_t record;
            record.thread_id    = thread_idx;
            record.sequence_num = i;
            record.global_seq   = global_seq.fetch_add(1, std::memory_order_acq_rel);
            record.timestamp    = get_timestamp_ns();

            bool success = buffer_v->emplace(1, 1, record);
            EXPECT_TRUE(success) << "Failed to emplace record: thread=" << thread_idx
                                 << " seq=" << i;

            // Small random delay to interleave threads
            if(i % 10 == 0)
            {
                std::this_thread::yield();
            }
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(writer_func, t);
    }

    // Wait for all writers
    for(auto& t : threads)
    {
        t.join();
    }

    // Final flush
    auto flush_status = buffer::flush(*buffer_id, true);
    EXPECT_EQ(flush_status, ROCPROFILER_STATUS_SUCCESS);

    // Wait a bit for async processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Verify results
    const size_t expected_records = num_threads * records_per_thread;

    // Check record count - data loss detection
    EXPECT_EQ(callback_data.records.size(), expected_records)
        << "Record count mismatch: expected=" << expected_records
        << " received=" << callback_data.records.size() << " (data loss detected!)";

    // Group records by thread
    std::vector<std::vector<test_record_t>> per_thread_records(num_threads);
    for(const auto& record : callback_data.records)
    {
        if(record.thread_id < num_threads)
        {
            per_thread_records[record.thread_id].push_back(record);
        }
    }

    // Check each thread got the expected number of records
    for(size_t t = 0; t < num_threads; ++t)
    {
        EXPECT_EQ(per_thread_records[t].size(), records_per_thread)
            << "Thread " << t << " record count mismatch";
    }

    // CRITICAL CHECK: Verify per-thread ordering is preserved
    // In the broken implementation, buffer flush order is not FIFO,
    // so sequence numbers may be received out of order
    size_t ordering_violations = 0;
    for(size_t t = 0; t < num_threads; ++t)
    {
        auto& thread_records = per_thread_records[t];
        // Sort by the order they were received (their index in the callback)
        // Check that sequence numbers are monotonically increasing
        for(size_t i = 1; i < thread_records.size(); ++i)
        {
            if(thread_records[i].sequence_num < thread_records[i - 1].sequence_num)
            {
                ordering_violations++;
                std::cerr << "ORDERING VIOLATION: Thread " << t << " received seq "
                          << thread_records[i].sequence_num << " after seq "
                          << thread_records[i - 1].sequence_num << std::endl;
            }
        }
    }

    EXPECT_EQ(ordering_violations, 0)
        << "Found " << ordering_violations << " per-thread ordering violations "
        << "(FIFO flush ordering bug detected!)";

    // Cleanup
    auto destroy_status = rocprofiler_destroy_buffer(*buffer_id);
    EXPECT_EQ(destroy_status, ROCPROFILER_STATUS_SUCCESS);
}

/**
 * @test RapidRotationStressTest
 * @brief Test rapid buffer rotations don't cause data loss or corruption.
 *
 * This test uses very small buffers and high write rates to stress the
 * buffer rotation logic. The old implementation may reuse a buffer that
 * is still being flushed when both buffers fill up rapidly.
 */
TEST(buffer_ordering_stress, rapid_rotation_completeness)
{
    namespace buffer = ::rocprofiler::buffer;

    constexpr size_t num_threads        = 8;
    constexpr size_t records_per_thread = 200;
    constexpr size_t buffer_size        = 512;  // Very small buffer
    constexpr size_t watermark          = 2;    // Flush after just 2 records

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

    // Track all records written
    std::atomic<uint64_t>          records_written{0};
    std::vector<std::atomic<bool>> record_seen(num_threads * records_per_thread);
    for(auto& seen : record_seen)
    {
        seen.store(false, std::memory_order_relaxed);
    }

    auto writer_func = [&](size_t thread_idx) {
        for(size_t i = 0; i < records_per_thread; ++i)
        {
            test_record_t record;
            record.thread_id    = thread_idx;
            record.sequence_num = i;
            record.global_seq   = thread_idx * records_per_thread + i;  // Unique ID
            record.timestamp    = get_timestamp_ns();

            bool success = buffer_v->emplace(1, 1, record);
            if(success)
            {
                records_written.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::vector<std::thread> threads;
    for(size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(writer_func, t);
    }

    for(auto& t : threads)
    {
        t.join();
    }

    buffer::flush(*buffer_id, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    const size_t expected = num_threads * records_per_thread;
    const size_t received = callback_data.records.size();

    std::cerr << "Records written: " << records_written.load() << ", Records received: " << received
              << ", Expected: " << expected << std::endl;

    // Check completeness - no data loss
    EXPECT_EQ(received, expected) << "Data loss detected: expected=" << expected
                                  << " received=" << received << " difference="
                                  << (expected > received ? expected - received
                                                          : received - expected);

    // Check for duplicates
    std::set<uint64_t> seen_ids;
    size_t             duplicates = 0;
    for(const auto& record : callback_data.records)
    {
        if(seen_ids.count(record.global_seq) > 0)
        {
            duplicates++;
        }
        seen_ids.insert(record.global_seq);
    }

    EXPECT_EQ(duplicates, 0) << "Found " << duplicates << " duplicate records";

    rocprofiler_destroy_buffer(*buffer_id);
}

/**
 * @test FlushDuringWriteTest
 * @brief Test that concurrent flush and write operations don't corrupt data.
 *
 * The old implementation's syncer flags may not properly protect against
 * writes occurring during flush callback execution.
 */
TEST(buffer_ordering_stress, flush_during_write_integrity)
{
    namespace buffer = ::rocprofiler::buffer;

    constexpr size_t buffer_size   = 2048;
    constexpr size_t watermark     = 10;
    constexpr size_t total_records = 1000;
    constexpr size_t num_writers   = 4;

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

    std::atomic<bool>     stop_flushing{false};
    std::atomic<uint64_t> flush_count{0};

    // Flusher thread: periodically flush while writers are active
    auto flusher_func = [&]() {
        while(!stop_flushing.load(std::memory_order_acquire))
        {
            buffer::flush(*buffer_id, false);
            flush_count.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    };

    std::atomic<uint64_t> write_count{0};

    auto writer_func = [&](size_t thread_idx) {
        const size_t records_to_write = total_records / num_writers;
        for(size_t i = 0; i < records_to_write; ++i)
        {
            test_record_t record;
            record.thread_id    = thread_idx;
            record.sequence_num = i;
            record.global_seq   = write_count.fetch_add(1, std::memory_order_acq_rel);
            record.timestamp    = get_timestamp_ns();

            buffer_v->emplace(1, 1, record);
        }
    };

    // Start flusher
    std::thread flusher(flusher_func);

    // Start writers
    std::vector<std::thread> writers;
    for(size_t t = 0; t < num_writers; ++t)
    {
        writers.emplace_back(writer_func, t);
    }

    // Wait for writers
    for(auto& w : writers)
    {
        w.join();
    }

    // Stop flusher
    stop_flushing.store(true, std::memory_order_release);
    flusher.join();

    // Final flush
    buffer::flush(*buffer_id, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cerr << "Flush operations: " << flush_count.load()
              << ", Records received: " << callback_data.records.size()
              << ", Expected: " << total_records << std::endl;

    // Verify data integrity - no records lost during concurrent flush/write
    EXPECT_EQ(callback_data.records.size(), total_records)
        << "Data integrity issue: concurrent flush/write caused record loss";

    // Verify no corruption - check record values are valid
    for(const auto& record : callback_data.records)
    {
        EXPECT_LT(record.thread_id, num_writers)
            << "Corrupted thread_id detected: " << record.thread_id;
        EXPECT_LT(record.global_seq, total_records)
            << "Corrupted global_seq detected: " << record.global_seq;
    }

    rocprofiler_destroy_buffer(*buffer_id);
}

/**
 * @test GlobalOrderingTest
 * @brief Verify that global record ordering respects write order across flushes.
 *
 * Single-threaded test to verify that records written in sequence A, B, C
 * are received in sequence A, B, C even across multiple buffer flushes.
 */
TEST(buffer_ordering_stress, global_fifo_ordering)
{
    namespace buffer = ::rocprofiler::buffer;

    constexpr size_t buffer_size   = 256;  // Very small to force many rotations
    constexpr size_t watermark     = 2;    // Flush very frequently
    constexpr size_t total_records = 100;

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

    // Write records sequentially with known order
    for(size_t i = 0; i < total_records; ++i)
    {
        test_record_t record;
        record.thread_id    = 0;
        record.sequence_num = i;
        record.global_seq   = i;
        record.timestamp    = get_timestamp_ns();

        bool success = buffer_v->emplace(1, 1, record);
        EXPECT_TRUE(success) << "Failed to emplace record " << i;
    }

    buffer::flush(*buffer_id, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Verify count
    EXPECT_EQ(callback_data.records.size(), total_records)
        << "Expected " << total_records << " records, got " << callback_data.records.size();

    // CRITICAL: Verify FIFO ordering is preserved
    // Records should be received in the same order they were written
    size_t order_errors = 0;
    for(size_t i = 0; i < callback_data.records.size(); ++i)
    {
        if(callback_data.records[i].global_seq != i)
        {
            order_errors++;
            if(order_errors <= 10)  // Limit error output
            {
                std::cerr << "GLOBAL ORDERING ERROR at position " << i
                          << ": expected global_seq=" << i
                          << " but got global_seq=" << callback_data.records[i].global_seq
                          << std::endl;
            }
        }
    }

    EXPECT_EQ(order_errors, 0) << "Found " << order_errors << " global ordering errors "
                               << "(buffers flushed out of FIFO order!)";

    rocprofiler_destroy_buffer(*buffer_id);
}
