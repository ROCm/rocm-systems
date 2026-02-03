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

#include "../source/record_collector.hpp"
#include "../source/types.hpp"

#include <gtest/gtest.h>

#include <string>
#include <thread>
#include <vector>

namespace rocprofiler
{
namespace python
{
namespace test
{
/**
 * @brief Tests for RecordCollector class
 *
 * Note: These tests focus on the thread-safe storage and retrieval of records.
 * Full integration tests with rocprofiler-sdk require GPU hardware.
 */
class RecordCollectorTest : public ::testing::Test
{
protected:
    void SetUp() override { collector_ = std::make_unique<RecordCollector>(); }

    void TearDown() override { collector_.reset(); }

    std::unique_ptr<RecordCollector> collector_;
};

TEST_F(RecordCollectorTest, DefaultConstruction)
{
    EXPECT_EQ(collector_->record_count(), 0);
    EXPECT_TRUE(collector_->get_records().empty());
}

TEST_F(RecordCollectorTest, ClearEmptyCollector)
{
    // Should not throw on empty collector
    EXPECT_NO_THROW(collector_->clear());
    EXPECT_EQ(collector_->record_count(), 0);
}

TEST_F(RecordCollectorTest, GetRecordsReturnsEmpty)
{
    auto records = collector_->get_records();
    EXPECT_TRUE(records.empty());
}

TEST_F(RecordCollectorTest, NonCopyable)
{
    // Verify RecordCollector is non-copyable (compilation test)
    EXPECT_FALSE(std::is_copy_constructible<RecordCollector>::value);
    EXPECT_FALSE(std::is_copy_assignable<RecordCollector>::value);
}

TEST_F(RecordCollectorTest, NonMovable)
{
    // Verify RecordCollector is non-movable (compilation test)
    EXPECT_FALSE(std::is_move_constructible<RecordCollector>::value);
    EXPECT_FALSE(std::is_move_assignable<RecordCollector>::value);
}

/**
 * @brief Test thread safety of RecordCollector
 *
 * This test creates multiple threads that call get_records() and clear()
 * concurrently to verify there are no data races.
 */
TEST_F(RecordCollectorTest, ThreadSafetyReadOperations)
{
    constexpr int num_threads   = 4;
    constexpr int num_reads     = 100;
    auto          collector_ptr = collector_.get();

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for(int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back([collector_ptr, num_reads]() {
            for(int j = 0; j < num_reads; ++j)
            {
                auto records = collector_ptr->get_records();
                auto count   = collector_ptr->record_count();
                (void) records;
                (void) count;
            }
        });
    }

    for(auto& t : threads)
    {
        t.join();
    }

    // If we get here without crashes or hangs, the thread safety test passed
    SUCCEED();
}

/**
 * @brief Test concurrent clear operations
 */
TEST_F(RecordCollectorTest, ThreadSafetyClearOperations)
{
    constexpr int num_threads   = 4;
    constexpr int num_clears    = 50;
    auto          collector_ptr = collector_.get();

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for(int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back([collector_ptr, num_clears]() {
            for(int j = 0; j < num_clears; ++j)
            {
                collector_ptr->clear();
                auto records = collector_ptr->get_records();
                (void) records;
            }
        });
    }

    for(auto& t : threads)
    {
        t.join();
    }

    SUCCEED();
}

/**
 * @brief Tests for RecordCollector dispatch header and counter record handling
 *
 * Note: These tests would require mocking rocprofiler types or using
 * the actual rocprofiler-sdk, which requires GPU hardware. The tests
 * above cover the basic functionality that can be tested without hardware.
 */

}  // namespace test
}  // namespace python
}  // namespace rocprofiler
