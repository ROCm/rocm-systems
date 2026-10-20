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

#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/units.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <gtest/gtest.h>

#include <pthread.h>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <typeinfo>
#include <vector>

TEST(rocprofiler_lib, buffer)
{
    namespace buffer = ::rocprofiler::buffer;
    namespace common = ::rocprofiler::common;

    ASSERT_EQ(buffer::get_buffers()->size(), 0)
        << "no buffers should have been created at this point";
    auto buffer_id = buffer::allocate_buffer();

    EXPECT_TRUE(buffer_id) << "failed to allocate buffer";
    EXPECT_GT(buffer_id->handle, 0);
    EXPECT_TRUE(buffer::is_valid_buffer_id(*buffer_id)) << "id=" << buffer_id->handle;
    ASSERT_EQ(buffer::get_buffers()->size(), 1) << "incorrect number of buffers created";

    // get pointer to buffer
    auto* buffer_v = buffer::get_buffer(*buffer_id);
    ASSERT_NE(buffer_v, nullptr) << "get_buffer returned a nullptr. id=" << buffer_id->handle;
    EXPECT_EQ(buffer_v->buffer_id, buffer_id->handle);

    buffer_v->watermark = common::units::get_page_size();
    EXPECT_EQ(buffer_v->get_internal_buffer().get_num_record_headers(), 0);

    EXPECT_TRUE(buffer_v->get_internal_buffer().allocate(sizeof(rocprofiler_buffer_id_t)));

    EXPECT_EQ(buffer_v->get_internal_buffer().capacity(), common::units::get_page_size());

    auto data = *buffer_id;
    buffer_v->emplace(1, 1, data);

    EXPECT_EQ(buffer_v->get_internal_buffer().get_num_record_headers(), 1);

    auto flush_status = buffer::flush(*buffer_id, true);
    EXPECT_EQ(flush_status, ROCPROFILER_STATUS_SUCCESS);

    auto destroy_status = rocprofiler_destroy_buffer(*buffer_id);
    EXPECT_EQ(destroy_status, ROCPROFILER_STATUS_SUCCESS);
}

namespace
{
struct buffer_records
{
    std::vector<uint64_t> values     = {};
    uint64_t              drop_count = 0;
    size_t                callbacks  = 0;
};

void
collect_records(rocprofiler_context_id_t,
                rocprofiler_buffer_id_t,
                rocprofiler_record_header_t** headers,
                size_t                        num_headers,
                void*                         data,
                uint64_t                      drop_count)
{
    auto& records = *static_cast<buffer_records*>(data);
    for(size_t i = 0; i < num_headers; ++i)
    {
        EXPECT_EQ(headers[i]->category, 1);
        EXPECT_EQ(headers[i]->kind, 1);
        records.values.emplace_back(*static_cast<uint64_t*>(headers[i]->payload));
    }
    records.drop_count = drop_count;
    ++records.callbacks;
}

rocprofiler_status_t
create_test_buffer(size_t                      size,
                   size_t                      watermark,
                   rocprofiler_buffer_policy_t policy,
                   buffer_records&             records,
                   rocprofiler_buffer_id_t&    buffer_id)
{
    // Buffer creation is only allowed during client initialization.
    namespace registration    = rocprofiler::registration;
    const auto init_status    = registration::get_init_status();
    auto       restore_status = rocprofiler::common::scope_destructor{
        [init_status]() { registration::set_init_status(init_status); }};
    registration::set_init_status(-1);
    return rocprofiler_create_buffer(
        {}, size, watermark, policy, collect_records, &records, &buffer_id);
}

void
test_buffer_rollover(rocprofiler_buffer_policy_t policy, bool automatic_flush)
{
    auto records   = buffer_records{};
    auto buffer_id = rocprofiler_buffer_id_t{};
    auto size      = rocprofiler::common::units::get_page_size();
    ASSERT_EQ(create_test_buffer(size, automatic_flush ? 0 : size, policy, records, buffer_id),
              ROCPROFILER_STATUS_SUCCESS);

    auto* buffer = rocprofiler::buffer::get_buffer(buffer_id);
    ASSERT_NE(buffer, nullptr);

    // Traverse both internal buffers more than once. With a zero watermark, each
    // insertion schedules an asynchronous flush and selects the next buffer.
    for(uint64_t value = 0; value < 4; ++value)
    {
        EXPECT_TRUE(buffer->emplace(1, 1, value)) << "record " << value;
        if(!automatic_flush)
        {
            EXPECT_EQ(rocprofiler::buffer::flush(buffer_id, true), ROCPROFILER_STATUS_SUCCESS);
        }
    }

    // Wait for all callbacks before inspecting their output or destroying the buffer.
    EXPECT_EQ(rocprofiler::buffer::flush(buffer_id, true), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(records.values, (std::vector<uint64_t>{0, 1, 2, 3}));
    EXPECT_EQ(records.callbacks, 4);
    EXPECT_EQ(records.drop_count, 0);
    EXPECT_EQ(rocprofiler_destroy_buffer(buffer_id), ROCPROFILER_STATUS_SUCCESS);
}
}  // namespace

TEST(rocprofiler_lib, buffer_discard_explicit_flush)
{
    test_buffer_rollover(ROCPROFILER_BUFFER_POLICY_DISCARD, false);
}

TEST(rocprofiler_lib, buffer_discard_watermark_flush)
{
    test_buffer_rollover(ROCPROFILER_BUFFER_POLICY_DISCARD, true);
}

TEST(rocprofiler_lib, buffer_lossless_explicit_flush)
{
    test_buffer_rollover(ROCPROFILER_BUFFER_POLICY_LOSSLESS, false);
}

TEST(rocprofiler_lib, buffer_lossless_watermark_flush)
{
    test_buffer_rollover(ROCPROFILER_BUFFER_POLICY_LOSSLESS, true);
}

TEST(rocprofiler_lib, buffer_discard_full_recovery)
{
    auto records   = buffer_records{};
    auto buffer_id = rocprofiler_buffer_id_t{};
    auto size      = rocprofiler::common::units::get_page_size();
    // Keep the watermark above capacity to exercise dropping records rather than
    // automatic flushing.
    ASSERT_EQ(
        create_test_buffer(size, 2 * size, ROCPROFILER_BUFFER_POLICY_DISCARD, records, buffer_id),
        ROCPROFILER_STATUS_SUCCESS);
    auto* buffer = rocprofiler::buffer::get_buffer(buffer_id);
    ASSERT_NE(buffer, nullptr);

    auto     expected = std::vector<uint64_t>{};
    uint64_t value    = 0;
    for(size_t cycle = 0; cycle < 4; ++cycle)
    {
        const auto capacity = buffer->get_internal_buffer().capacity() / sizeof(value);
        EXPECT_GT(capacity, 0);
        for(size_t i = 0; i < capacity; ++i)
        {
            EXPECT_TRUE(buffer->emplace(1, 1, value));
            expected.emplace_back(value++);
        }
        EXPECT_FALSE(buffer->emplace(1, 1, value));
        EXPECT_EQ(rocprofiler::buffer::flush(buffer_id, true), ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(records.values, expected);
        EXPECT_EQ(records.callbacks, cycle + 1);
        EXPECT_EQ(records.drop_count, cycle + 1);
    }
    EXPECT_EQ(rocprofiler_destroy_buffer(buffer_id), ROCPROFILER_STATUS_SUCCESS);
}
