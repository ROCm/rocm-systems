// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "counter_info.hpp"
#include "buffered_output.hpp"
#include "tmp_file_buffer.hpp"

#include "lib/common/container/ring_buffer.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

#include <string_view>
#include <unordered_set>

namespace rocprofiler
{
namespace tool
{
constexpr auto type = domain_type::COUNTER_VALUES;

std::vector<tool_counter_value_t>
tool_counter_record_t::getRecords() const
{
    auto& _tmp_file = get_tmp_file_buffer<tool_counter_value_t>(type)->file;

    return _tmp_file.read<tool_counter_value_t>(records.offset, records.count);
}

void
tool_counter_record_t::writeRecord(const tool_counter_value_t* ptr, size_t num_records)
{
    auto& _tmp_file = get_tmp_file_buffer<tool_counter_value_t>(type)->file;

    records.offset = _tmp_file.write<tool_counter_value_t>(ptr, num_records);
    records.count  = num_records;
}
}  // namespace tool
}  // namespace rocprofiler
