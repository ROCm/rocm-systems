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
#include <cstdint>
#include <stdint.h>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#if ROCPROFSYS_USE_ROCM > 0
#    include <rocprofiler-sdk/version.h>
#endif

namespace rocprofsys
{
namespace trace_cache
{

struct storage_parsed_type_base
{};

struct kernel_dispatch_sample : storage_parsed_type_base
{
    // Timing fields
    uint64_t start_timestamp;
    uint64_t end_timestamp;

    // Identification fields
    uint64_t thread_id;
    uint64_t agent_id_handle;
    uint64_t kernel_id;
    uint64_t dispatch_id;
    uint64_t queue_id_handle;

    // Correlation fields
    uint64_t correlation_id_internal;
    uint64_t correlation_id_ancestor;

    // Dispatch configuration
    uint32_t private_segment_size;
    uint32_t group_segment_size;
    uint32_t workgroup_size_x;
    uint32_t workgroup_size_y;
    uint32_t workgroup_size_z;
    uint32_t grid_size_x;
    uint32_t grid_size_y;
    uint32_t grid_size_z;

    // Stream handle
    size_t stream_handle;
};

struct memory_copy_sample : storage_parsed_type_base
{
    // Timing fields
    uint64_t start_timestamp;
    uint64_t end_timestamp;

    // Identification fields
    uint64_t thread_id;
    uint64_t dst_agent_id_handle;
    uint64_t src_agent_id_handle;

    // Operation details
    int32_t  kind;
    int32_t  operation;
    uint64_t bytes;

    // Correlation fields
    uint64_t correlation_id_internal;
    uint64_t correlation_id_ancestor;

    // Address fields (version dependent)
    uint64_t dst_address_value;
    uint64_t src_address_value;

    // Stream handle
    size_t stream_handle;
};

#if(ROCPROFILER_VERSION >= 600)
struct memory_allocate_sample : storage_parsed_type_base
{
    // Timing fields
    uint64_t start_timestamp;
    uint64_t end_timestamp;

    // Identification fields
    uint64_t thread_id;
    uint64_t agent_id_handle;

    // Operation details
    int32_t  kind;
    int32_t  operation;
    uint64_t allocation_size;

    // Correlation fields
    uint64_t correlation_id_internal;
    uint64_t correlation_id_ancestor;

    // Address fields (version dependent)
    uint64_t address_value;

    // Stream handle
    size_t stream_handle;
};
#endif

struct region_sample : storage_parsed_type_base
{
    region_sample() = default;
    region_sample(uint64_t _thread_id, std::string _name,
                  uint64_t _correlation_id_internal, uint64_t _correlation_id_ancestor,
                  uint64_t _start_timestamp, uint64_t _end_timestamp,
                  std::string _call_stack, std::string _args_str, std::string _category)
    : thread_id(_thread_id)
    , name(std::move(_name))
    , correlation_id_internal(_correlation_id_internal)
    , correlation_id_ancestor(_correlation_id_ancestor)
    , start_timestamp(_start_timestamp)
    , end_timestamp(_end_timestamp)
    , call_stack(std::move(_call_stack))
    , args_str(std::move(_args_str))
    , category(std::move(_category))
    {}

    uint64_t    thread_id;
    std::string name;

    uint64_t correlation_id_internal;
    uint64_t correlation_id_ancestor;

    uint64_t start_timestamp;
    uint64_t end_timestamp;

    std::string call_stack;
    std::string args_str;
    std::string category;
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
    uint32_t    device_id;
    uint8_t     device_type;
    std::string pmc_info_name;
    double      value;
};

struct amd_smi_sample : storage_parsed_type_base
{
    enum class settings_positions : uint8_t
    {
        busy = 0,
        temp,
        power,
        mem_usage,
        vcn_activity,
        jpeg_activity
    };

    uint64_t             settings;  // bitfield
    uint32_t             device_id;
    size_t               timestamp;
    uint32_t             gfx_activity;
    uint32_t             umc_activity;
    uint32_t             mm_activity;
    uint32_t             power;
    int64_t              temperature;
    size_t               mem_usage;
    std::vector<uint8_t> xcp_activity;
};

struct cpu_freq_sample : storage_parsed_type_base
{
    size_t               timestamp;
    int64_t              page_rss;
    int64_t              virt_mem_usage;
    int64_t              peak_rss;
    int64_t              context_switch_count;
    int64_t              page_faults;
    int64_t              user_mode_time;
    int64_t              kernel_mode_time;
    std::vector<uint8_t> freqs;
};

struct backtrace_region_sample : storage_parsed_type_base
{
    backtrace_region_sample() = default;
    backtrace_region_sample(uint32_t _type, uint64_t _thread_id, std::string _track_name,
                            std::string _name, uint64_t _start_timestamp,
                            uint64_t _end_timestamp, std::string _category,
                            std::string _call_stack, std::string _line_info,
                            std::string _extdata)
    : type(_type)
    , thread_id(_thread_id)
    , track_name(std::move(_track_name))
    , name(std::move(_name))
    , start_timestamp(_start_timestamp)
    , end_timestamp(_end_timestamp)
    , category(std::move(_category))
    , call_stack(std::move(_call_stack))
    , line_info(std::move(_line_info))
    , extdata(std::move(_extdata))
    {}

    uint32_t    type;
    uint64_t    thread_id;
    std::string track_name;
    std::string name;

    uint64_t start_timestamp;
    uint64_t end_timestamp;

    std::string category;
    std::string call_stack;
    std::string line_info;
    std::string extdata;
};

enum class entry_type : uint32_t
{
    in_time_sample        = 0x0000,
    pmc_event_with_sample = 0x0001,
    region                = 0x0002,
    kernel_dispatch       = 0x0003,
    memory_copy           = 0x0004,
#if(ROCPROFSYS_USE_ROCM && ROCPROFILER_VERSION >= 600)
    memory_alloc = 0x0005,
#endif
    amd_smi_sample          = 0x0006,
    cpu_freq_sample         = 0x0007,
    backtrace_region_sample = 0x0008,
    fragmented_space        = 0xFFFF
};
}  // namespace trace_cache
}  // namespace rocprofsys
