// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <stdint.h>
#include <string>
#include <unistd.h>
#include <utility>

namespace rocprofsys
{
namespace sample_cache
{

struct storage_parsed_type_base
{};

struct kernel_dispatch_sample : storage_parsed_type_base
{
    rocprofiler_buffer_tracing_kernel_dispatch_record_t record;
    size_t                                              stream_handle;
};

struct memory_copy_sample : storage_parsed_type_base
{
    rocprofiler_buffer_tracing_memory_copy_record_t record;
    size_t                                          stream_handle;
};

#if(ROCPROFILER_VERSION >= 600)
struct memory_allocate_sample : storage_parsed_type_base
{
    rocprofiler_buffer_tracing_memory_allocation_record_t record;
    size_t                                                stream_handle;
};
#endif

struct region_sample : storage_parsed_type_base
{
    region_sample() = default;
    region_sample(rocprofiler_callback_tracing_record_t _record,
                  rocprofiler_timestamp_t               _start_timestamp,
                  rocprofiler_timestamp_t _end_timestamp, std::string _call_stack,
                  std::string _args_str, std::string _category)
    : record(_record)
    , start_timestamp(_start_timestamp)
    , end_timestamp(_end_timestamp)
    , call_stack(std::move(_call_stack))
    , args_str(std::move(_args_str))
    , category(std::move(_category))
    {}
    rocprofiler_callback_tracing_record_t record;
    rocprofiler_timestamp_t               start_timestamp;
    rocprofiler_timestamp_t               end_timestamp;
    std::string                           call_stack;
    std::string                           args_str;
    std::string                           category;
};

struct in_time_sample : storage_parsed_type_base
{
    std::string track_name;
    size_t      timestamp_ns;
    std::string event_metadata;
    size_t      stack_id;
    size_t      parent_stack_id;
    size_t      correlation_id;
    std::string call_stack;
    std::string line_info;
};

struct pmc_event_with_sample : in_time_sample
{
    size_t      agent_handle;
    std::string pmc_info_name;
    size_t      value;
};

enum class entry_type : uint32_t
{
    in_time_sample        = 0x0000,
    pmc_event_with_sample = 0x0001,
    region                = 0x0002,
    kernel_dispatch       = 0x0003,
    memory_copy           = 0x0004,
#if(ROCPROFILER_VERSION >= 600)
    memory_alloc = 0x0005,
#endif
    fragmented_space = 0xFFFF
};
}  // namespace sample_cache
}  // namespace rocprofsys
