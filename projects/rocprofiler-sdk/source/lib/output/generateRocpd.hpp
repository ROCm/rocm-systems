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

#pragma once

#include "agent_info.hpp"
#include "generator.hpp"
#include "lib/output/sql/bind_data_type.hpp"
#include "lib/output/sql/common.hpp"
#include "metadata.hpp"
#include "output_config.hpp"
#include "stream_info.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <sqlite3.h>

#include <cstdint>
#include <deque>
#include <functional>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>

namespace rocprofiler
{
namespace tool
{
void
write_rocpd(
    const output_config&                                                    cfg,
    const metadata&                                                         tool_metadata,
    const std::vector<agent_info>&                                          agent_data,
    const generator<tool_buffer_tracing_hip_api_ext_record_t>&              hip_api_gen,
    const generator<rocprofiler_buffer_tracing_hsa_api_record_t>&           hsa_api_gen,
    const generator<tool_buffer_tracing_kernel_dispatch_ext_record_t>&      kernel_dispatch_gen,
    const generator<tool_buffer_tracing_memory_copy_ext_record_t>&          memory_copy_gen,
    const generator<rocprofiler_buffer_tracing_marker_api_record_t>&        marker_api_gen,
    const generator<tool_buffer_tracing_memory_allocation_ext_record_t>&    memory_alloc_gen,
    const generator<rocprofiler_buffer_tracing_scratch_memory_record_t>&    scratch_memory_gen,
    const generator<rocprofiler_buffer_tracing_rccl_api_record_t>&          rccl_api_gen,
    const generator<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>& rocdecode_api_gen,
    const generator<tool_counter_record_t>&                                 counter_collection_gen);

// used in schema generation
struct argument_info
{
    uint32_t    arg_number = 0;
    std::string arg_type   = {};
    std::string arg_name   = {};
    std::string arg_value  = {};
};

struct sql_insert_value
{
    using binder_type = sql::binder_func_t;

    std::string_view name   = {};
    std::string_view value  = {};
    binder_type      binder = {};

    template <typename Tp>
    static binder_type make_binder(Tp&& value)
    {
        using value_type = common::mpl::unqualified_type_t<Tp>;
        return [val = std::forward<Tp>(value)](sqlite3_stmt* stmt, int32_t col) -> int {
            return sql::bind_data_type<value_type>{}(stmt, col, val);
        };
    }
};

struct track_data
{
    uint64_t                   nid       = 0;
    std::optional<pid_t>       ppid      = {};
    std::optional<pid_t>       pid       = {};
    std::optional<pid_t>       tid       = {};
    std::optional<uint64_t>    agent_id  = {};
    std::optional<uint64_t>    queue_id  = {};
    std::optional<uint64_t>    stream_id = {};
    std::optional<uint64_t>    name_id   = {};
    std::optional<std::string> extdata   = {};

    size_t         hash() const;
    decltype(auto) tie() const;

    std::vector<sql_insert_value> get_insert_values() const;
};

inline decltype(auto)
track_data::tie() const
{
    return std::tie(nid, ppid, pid, tid, agent_id, queue_id, stream_id);
}

bool
operator==(const track_data& lhs, const track_data& rhs);
}  // namespace tool
}  // namespace rocprofiler
