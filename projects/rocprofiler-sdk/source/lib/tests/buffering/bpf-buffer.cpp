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

#include "lib/common/container/bpf_record_header_buffer.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

namespace
{
using bpf_record_header_buffer_t = rocprofiler::common::container::bpf_record_header_buffer;

struct test_payload
{
    uint64_t value = 0;
};
}  // namespace

TEST(bpf_buffer, zero_hash_records_are_delivered)
{
    auto buffer  = bpf_record_header_buffer_t{4096};
    auto payload = test_payload{42};

    ASSERT_TRUE(buffer.emplace(uint64_t{0}, payload));

    auto num_headers = buffer.process_record_headers(std::true_type{}, [](auto&& records) {
        ASSERT_EQ(records.size(), 1);
        auto* record = records.at(0);
        ASSERT_NE(record, nullptr);
        ASSERT_NE(record->payload, nullptr);
        EXPECT_EQ(record->hash, 0);
        EXPECT_EQ(static_cast<test_payload*>(record->payload)->value, 42);
    });

    EXPECT_EQ(num_headers, 1);
    EXPECT_EQ(buffer.get_num_record_headers(), 0);
}
