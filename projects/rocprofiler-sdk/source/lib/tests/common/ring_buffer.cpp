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

#include "lib/common/container/record_header_buffer.hpp"
#include "lib/common/container/ring_buffer.hpp"
#include "lib/common/units.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace
{
namespace container = ::rocprofiler::common::container;
namespace units     = ::rocprofiler::common::units;
}  // namespace

TEST(ring_buffer, init_zero_size)
{
    // A zero-length request must not attempt a zero-size mmap() (EINVAL). It is
    // promoted to a minimum of one page.
    auto _buf = container::base::ring_buffer{};
    ASSERT_NO_FATAL_FAILURE(_buf.init(0));

    EXPECT_TRUE(_buf.is_initialized());
    EXPECT_NE(_buf.data(), nullptr);
    EXPECT_EQ(_buf.capacity(), units::get_page_size());
    EXPECT_TRUE(_buf.is_empty());
    EXPECT_FALSE(_buf.is_full());

    // the buffer is fully usable
    auto* _ptr = _buf.request(sizeof(uint64_t), alignof(uint64_t), false);
    EXPECT_NE(_ptr, nullptr);
    EXPECT_EQ(_buf.count(), sizeof(uint64_t));
}

TEST(ring_buffer, init_zero_size_constructor)
{
    auto _buf = container::base::ring_buffer{0};

    EXPECT_TRUE(_buf.is_initialized());
    EXPECT_EQ(_buf.capacity(), units::get_page_size());
}

TEST(ring_buffer, init_rounds_to_page_size)
{
    // non-zero sizes keep rounding up to a multiple of the page size
    const auto _page = units::get_page_size();

    auto _one = container::base::ring_buffer{1};
    EXPECT_EQ(_one.capacity(), _page);

    auto _exact = container::base::ring_buffer{_page};
    EXPECT_EQ(_exact.capacity(), _page);

    auto _over = container::base::ring_buffer{_page + 1};
    EXPECT_EQ(_over.capacity(), 2 * _page);
}

TEST(ring_buffer, record_header_buffer_allocate_zero_size)
{
    // record_header_buffer::allocate(0) is reached for an empty record domain
    auto _rhb = container::record_header_buffer{};
    ASSERT_NO_FATAL_FAILURE(_rhb.allocate(0));

    EXPECT_TRUE(_rhb.is_allocated());
    EXPECT_EQ(_rhb.capacity(), units::get_page_size());
    EXPECT_EQ(_rhb.get_num_record_headers(), 0);
}
