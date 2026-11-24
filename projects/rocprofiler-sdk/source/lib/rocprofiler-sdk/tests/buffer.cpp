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

#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/common/units.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <gtest/gtest.h>

#include <pthread.h>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <typeinfo>

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
