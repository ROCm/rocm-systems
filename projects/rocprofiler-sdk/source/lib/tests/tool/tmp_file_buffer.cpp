// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/output/tmp_file_buffer.hpp"
#include "lib/output/buffered_output.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <new>
#include <vector>

namespace
{
struct test_record
{
    uint64_t value = 0;
};

struct retry_record
{
    uint64_t value = 0;
};
}  // namespace

TEST(tmp_file_buffer, suspended_fork_output_does_not_offload_a_full_buffer)
{
    namespace tool = ::rocprofiler::tool;

    tool::get_tmp_file_name_callback() = [](domain_type) {
        return "/tmp/rocprofiler-sdk-tmp-file-buffer-test.dat";
    };

    auto*& filebuf = tool::get_tmp_file_buffer<test_record>(domain_type::MARKER);
    ASSERT_NE(filebuf, nullptr);

    auto record = test_record{};
    while(auto* slot = filebuf->buffer.request(false))
        new(slot) test_record{record};
    ASSERT_TRUE(filebuf->buffer.is_full());

    tool::suspend_fork_output();
    tool::prepare_fork_file_buffers();
    tool::write_ring_buffer(record, domain_type::MARKER);
    tool::parent_fork_file_buffers();
    tool::resume_fork_output();

    EXPECT_TRUE(filebuf->buffer.is_full());
}

TEST(tmp_file_buffer, snapshot_releases_registry_before_later_prepare_work)
{
    namespace tool = ::rocprofiler::tool;
    tool::prepare_fork_file_buffers();
    ASSERT_TRUE(tool::fork_file_buffers_unchanged());
    // Same-thread registration would deadlock if prepare retained the registry mutex.
    tool::register_file_buffer({[]() {}, []() {}});
    EXPECT_FALSE(tool::fork_file_buffers_unchanged());
    tool::parent_fork_file_buffers();
}

TEST(tmp_file_buffer, preserved_read_stream_reopens_for_append)
{
    namespace tool        = ::rocprofiler::tool;
    constexpr auto domain = domain_type::HIP;

    tool::get_tmp_file_name_callback() = [](domain_type) {
        return "/tmp/rocprofiler-sdk-tmp-file-buffer-retry-test.dat";
    };

    auto output = tool::buffered_output<retry_record, domain>{true};
    output.reset();
    tool::write_ring_buffer(retry_record{1}, domain);
    output.read();
    output.preserve();

    tool::write_ring_buffer(retry_record{2}, domain);
    output.read();

    auto values = std::vector<uint64_t>{};
    {
        auto gen = output.get_generator();
        for(auto block : gen)
            for(const auto& record : gen.get(block))
                values.emplace_back(record.value);
    }

    EXPECT_EQ(values, (std::vector<uint64_t>{1, 2}));
    output.reset();
}
