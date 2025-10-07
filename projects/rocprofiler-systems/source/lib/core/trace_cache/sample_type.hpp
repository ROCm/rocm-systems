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
#include "core/trace_cache/cacheable.hpp"

#include <cstddef>
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

enum class type_identifier_t : uint32_t
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

struct kernel_dispatch_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::kernel_dispatch;

    kernel_dispatch_sample() = default;
    kernel_dispatch_sample(uint64_t _start_timestamp, uint64_t _end_timestamp,
                           uint64_t _thread_id, uint64_t _agent_id_handle,
                           uint64_t _kernel_id, uint64_t _dispatch_id,
                           uint64_t _queue_id_handle, uint64_t _correlation_id_internal,
                           uint64_t _correlation_id_ancestor,
                           uint32_t _private_segment_size, uint32_t _group_segment_size,
                           uint32_t _workgroup_size_x, uint32_t _workgroup_size_y,
                           uint32_t _workgroup_size_z, uint32_t _grid_size_x,
                           uint32_t _grid_size_y, uint32_t _grid_size_z,
                           size_t _stream_handle)
    : start_timestamp(_start_timestamp)
    , end_timestamp(_end_timestamp)
    , thread_id(_thread_id)
    , agent_id_handle(_agent_id_handle)
    , kernel_id(_kernel_id)
    , dispatch_id(_dispatch_id)
    , queue_id_handle(_queue_id_handle)
    , correlation_id_internal(_correlation_id_internal)
    , correlation_id_ancestor(_correlation_id_ancestor)
    , private_segment_size(_private_segment_size)
    , group_segment_size(_group_segment_size)
    , workgroup_size_x(_workgroup_size_x)
    , workgroup_size_y(_workgroup_size_y)
    , workgroup_size_z(_workgroup_size_z)
    , grid_size_x(_grid_size_x)
    , grid_size_y(_grid_size_y)
    , grid_size_z(_grid_size_z)
    , stream_handle(_stream_handle)
    {}

    uint64_t start_timestamp;
    uint64_t end_timestamp;
    uint64_t thread_id;
    uint64_t agent_id_handle;
    uint64_t kernel_id;
    uint64_t dispatch_id;
    uint64_t queue_id_handle;
    uint64_t correlation_id_internal;
    uint64_t correlation_id_ancestor;
    uint32_t private_segment_size;
    uint32_t group_segment_size;
    uint32_t workgroup_size_x;
    uint32_t workgroup_size_y;
    uint32_t workgroup_size_z;
    uint32_t grid_size_x;
    uint32_t grid_size_y;
    uint32_t grid_size_z;
    size_t   stream_handle;
};

template <>
inline void
serialize(uint8_t* buffer, const kernel_dispatch_sample& value)
{
    size_t position = 0;
    utility::store_value(value.start_timestamp, buffer, position);
    utility::store_value(value.end_timestamp, buffer, position);
    utility::store_value(value.thread_id, buffer, position);
    utility::store_value(value.agent_id_handle, buffer, position);
    utility::store_value(value.kernel_id, buffer, position);
    utility::store_value(value.dispatch_id, buffer, position);
    utility::store_value(value.queue_id_handle, buffer, position);
    utility::store_value(value.correlation_id_internal, buffer, position);
    utility::store_value(value.correlation_id_ancestor, buffer, position);
    utility::store_value(value.private_segment_size, buffer, position);
    utility::store_value(value.group_segment_size, buffer, position);
    utility::store_value(value.workgroup_size_x, buffer, position);
    utility::store_value(value.workgroup_size_y, buffer, position);
    utility::store_value(value.workgroup_size_z, buffer, position);
    utility::store_value(value.grid_size_x, buffer, position);
    utility::store_value(value.grid_size_y, buffer, position);
    utility::store_value(value.grid_size_z, buffer, position);
    utility::store_value(value.stream_handle, buffer, position);
}

template <>
inline kernel_dispatch_sample
deserialize(uint8_t*& buffer)
{
    kernel_dispatch_sample value;
    utility::parse_value(value.start_timestamp, buffer);
    utility::parse_value(value.end_timestamp, buffer);
    utility::parse_value(value.thread_id, buffer);
    utility::parse_value(value.agent_id_handle, buffer);
    utility::parse_value(value.kernel_id, buffer);
    utility::parse_value(value.dispatch_id, buffer);
    utility::parse_value(value.queue_id_handle, buffer);
    utility::parse_value(value.correlation_id_internal, buffer);
    utility::parse_value(value.correlation_id_ancestor, buffer);
    utility::parse_value(value.private_segment_size, buffer);
    utility::parse_value(value.group_segment_size, buffer);
    utility::parse_value(value.workgroup_size_x, buffer);
    utility::parse_value(value.workgroup_size_y, buffer);
    utility::parse_value(value.workgroup_size_z, buffer);
    utility::parse_value(value.grid_size_x, buffer);
    utility::parse_value(value.grid_size_y, buffer);
    utility::parse_value(value.grid_size_z, buffer);
    utility::parse_value(value.stream_handle, buffer);
    return value;
}

template <>
inline size_t
get_size(const kernel_dispatch_sample& value)
{
    return utility::get_size_helper(value.start_timestamp) +
           utility::get_size_helper(value.end_timestamp) +
           utility::get_size_helper(value.thread_id) +
           utility::get_size_helper(value.agent_id_handle) +
           utility::get_size_helper(value.kernel_id) +
           utility::get_size_helper(value.dispatch_id) +
           utility::get_size_helper(value.queue_id_handle) +
           utility::get_size_helper(value.correlation_id_internal) +
           utility::get_size_helper(value.correlation_id_ancestor) +
           utility::get_size_helper(value.private_segment_size) +
           utility::get_size_helper(value.group_segment_size) +
           utility::get_size_helper(value.workgroup_size_x) +
           utility::get_size_helper(value.workgroup_size_y) +
           utility::get_size_helper(value.workgroup_size_z) +
           utility::get_size_helper(value.grid_size_x) +
           utility::get_size_helper(value.grid_size_y) +
           utility::get_size_helper(value.grid_size_z) +
           utility::get_size_helper(value.stream_handle);
}

struct memory_copy_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::memory_copy;

    memory_copy_sample() = default;
    memory_copy_sample(uint64_t _start_timestamp, uint64_t _end_timestamp,
                       uint64_t _thread_id, uint64_t _dst_agent_id_handle,
                       uint64_t _src_agent_id_handle, int32_t _kind, int32_t _operation,
                       uint64_t _bytes, uint64_t _correlation_id_internal,
                       uint64_t _correlation_id_ancestor, uint64_t _dst_address_value,
                       uint64_t _src_address_value, size_t _stream_handle)
    : start_timestamp(_start_timestamp)
    , end_timestamp(_end_timestamp)
    , thread_id(_thread_id)
    , dst_agent_id_handle(_dst_agent_id_handle)
    , src_agent_id_handle(_src_agent_id_handle)
    , kind(_kind)
    , operation(_operation)
    , bytes(_bytes)
    , correlation_id_internal(_correlation_id_internal)
    , correlation_id_ancestor(_correlation_id_ancestor)
    , dst_address_value(_dst_address_value)
    , src_address_value(_src_address_value)
    , stream_handle(_stream_handle)
    {}

    uint64_t start_timestamp;
    uint64_t end_timestamp;
    uint64_t thread_id;
    uint64_t dst_agent_id_handle;
    uint64_t src_agent_id_handle;
    int32_t  kind;
    int32_t  operation;
    uint64_t bytes;
    uint64_t correlation_id_internal;
    uint64_t correlation_id_ancestor;
    uint64_t dst_address_value;
    uint64_t src_address_value;
    size_t   stream_handle;
};

template <>
inline void
serialize(uint8_t* buffer, const memory_copy_sample& value)
{
    size_t position = 0;
    utility::store_value(value.start_timestamp, buffer, position);
    utility::store_value(value.end_timestamp, buffer, position);
    utility::store_value(value.thread_id, buffer, position);
    utility::store_value(value.dst_agent_id_handle, buffer, position);
    utility::store_value(value.src_agent_id_handle, buffer, position);
    utility::store_value(value.kind, buffer, position);
    utility::store_value(value.operation, buffer, position);
    utility::store_value(value.bytes, buffer, position);
    utility::store_value(value.correlation_id_internal, buffer, position);
    utility::store_value(value.correlation_id_ancestor, buffer, position);
    utility::store_value(value.dst_address_value, buffer, position);
    utility::store_value(value.src_address_value, buffer, position);
    utility::store_value(value.stream_handle, buffer, position);
}

template <>
inline memory_copy_sample
deserialize(uint8_t*& buffer)
{
    memory_copy_sample value;
    utility::parse_value(value.start_timestamp, buffer);
    utility::parse_value(value.end_timestamp, buffer);
    utility::parse_value(value.thread_id, buffer);
    utility::parse_value(value.dst_agent_id_handle, buffer);
    utility::parse_value(value.src_agent_id_handle, buffer);
    utility::parse_value(value.kind, buffer);
    utility::parse_value(value.operation, buffer);
    utility::parse_value(value.bytes, buffer);
    utility::parse_value(value.correlation_id_internal, buffer);
    utility::parse_value(value.correlation_id_ancestor, buffer);
    utility::parse_value(value.dst_address_value, buffer);
    utility::parse_value(value.src_address_value, buffer);
    utility::parse_value(value.stream_handle, buffer);
    return value;
}

template <>
inline size_t
get_size(const memory_copy_sample& value)
{
    return utility::get_size_helper(value.start_timestamp) +
           utility::get_size_helper(value.end_timestamp) +
           utility::get_size_helper(value.thread_id) +
           utility::get_size_helper(value.dst_agent_id_handle) +
           utility::get_size_helper(value.src_agent_id_handle) +
           utility::get_size_helper(value.kind) +
           utility::get_size_helper(value.operation) +
           utility::get_size_helper(value.bytes) +
           utility::get_size_helper(value.correlation_id_internal) +
           utility::get_size_helper(value.correlation_id_ancestor) +
           utility::get_size_helper(value.dst_address_value) +
           utility::get_size_helper(value.src_address_value) +
           utility::get_size_helper(value.stream_handle);
}

#if(ROCPROFILER_VERSION >= 600)
struct memory_allocate_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::memory_alloc;

    memory_allocate_sample() = default;
    memory_allocate_sample(uint64_t _start_timestamp, uint64_t _end_timestamp,
                           uint64_t _thread_id, uint64_t _agent_id_handle, int32_t _kind,
                           int32_t _operation, uint64_t _allocation_size,
                           uint64_t _correlation_id_internal,
                           uint64_t _correlation_id_ancestor, uint64_t _address_value,
                           size_t _stream_handle)
    : start_timestamp(_start_timestamp)
    , end_timestamp(_end_timestamp)
    , thread_id(_thread_id)
    , agent_id_handle(_agent_id_handle)
    , kind(_kind)
    , operation(_operation)
    , allocation_size(_allocation_size)
    , correlation_id_internal(_correlation_id_internal)
    , correlation_id_ancestor(_correlation_id_ancestor)
    , address_value(_address_value)
    , stream_handle(_stream_handle)
    {}

    uint64_t start_timestamp;
    uint64_t end_timestamp;
    uint64_t thread_id;
    uint64_t agent_id_handle;
    int32_t  kind;
    int32_t  operation;
    uint64_t allocation_size;
    uint64_t correlation_id_internal;
    uint64_t correlation_id_ancestor;
    uint64_t address_value;
    size_t   stream_handle;
};

template <>
inline void
serialize(uint8_t* buffer, const memory_allocate_sample& value)
{
    size_t position = 0;
    utility::store_value(value.start_timestamp, buffer, position);
    utility::store_value(value.end_timestamp, buffer, position);
    utility::store_value(value.thread_id, buffer, position);
    utility::store_value(value.agent_id_handle, buffer, position);
    utility::store_value(value.kind, buffer, position);
    utility::store_value(value.operation, buffer, position);
    utility::store_value(value.allocation_size, buffer, position);
    utility::store_value(value.correlation_id_internal, buffer, position);
    utility::store_value(value.correlation_id_ancestor, buffer, position);
    utility::store_value(value.address_value, buffer, position);
    utility::store_value(value.stream_handle, buffer, position);
}

template <>
inline memory_allocate_sample
deserialize(uint8_t*& buffer)
{
    memory_allocate_sample value;
    utility::parse_value(value.start_timestamp, buffer);
    utility::parse_value(value.end_timestamp, buffer);
    utility::parse_value(value.thread_id, buffer);
    utility::parse_value(value.agent_id_handle, buffer);
    utility::parse_value(value.kind, buffer);
    utility::parse_value(value.operation, buffer);
    utility::parse_value(value.allocation_size, buffer);
    utility::parse_value(value.correlation_id_internal, buffer);
    utility::parse_value(value.correlation_id_ancestor, buffer);
    utility::parse_value(value.address_value, buffer);
    utility::parse_value(value.stream_handle, buffer);
    return value;
}

template <>
inline size_t
get_size(const memory_allocate_sample& value)
{
    return utility::get_size_helper(value.start_timestamp) +
           utility::get_size_helper(value.end_timestamp) +
           utility::get_size_helper(value.thread_id) +
           utility::get_size_helper(value.agent_id_handle) +
           utility::get_size_helper(value.kind) +
           utility::get_size_helper(value.operation) +
           utility::get_size_helper(value.allocation_size) +
           utility::get_size_helper(value.correlation_id_internal) +
           utility::get_size_helper(value.correlation_id_ancestor) +
           utility::get_size_helper(value.address_value) +
           utility::get_size_helper(value.stream_handle);
}
#endif

struct region_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::region;

    region_sample() = default;
    region_sample(uint64_t _thread_id, int32_t _kind, int32_t _operation,
                  uint64_t _correlation_id_internal, uint64_t _correlation_id_ancestor,
                  uint64_t _start_timestamp, uint64_t _end_timestamp,
                  std::string _call_stack, std::string _args_str, std::string _category)
    : thread_id(_thread_id)
    , kind(_kind)
    , operation(_operation)
    , correlation_id_internal(_correlation_id_internal)
    , correlation_id_ancestor(_correlation_id_ancestor)
    , start_timestamp(_start_timestamp)
    , end_timestamp(_end_timestamp)
    , call_stack(std::move(_call_stack))
    , args_str(std::move(_args_str))
    , category(std::move(_category))
    {}

    uint64_t thread_id;
    int32_t  kind;
    int32_t  operation;

    uint64_t correlation_id_internal;
    uint64_t correlation_id_ancestor;

    uint64_t start_timestamp;
    uint64_t end_timestamp;

    std::string call_stack;
    std::string args_str;
    std::string category;
};

template <>
inline void
serialize(uint8_t* buffer, const region_sample& value)
{
    size_t position = 0;
    utility::store_value(value.thread_id, buffer, position);
    utility::store_value(value.kind, buffer, position);
    utility::store_value(value.operation, buffer, position);
    utility::store_value(value.correlation_id_internal, buffer, position);
    utility::store_value(value.correlation_id_ancestor, buffer, position);
    utility::store_value(value.start_timestamp, buffer, position);
    utility::store_value(value.end_timestamp, buffer, position);
    utility::store_value(value.call_stack.c_str(), buffer, position);
    utility::store_value(value.args_str.c_str(), buffer, position);
    utility::store_value(value.category.c_str(), buffer, position);
}

template <>
inline region_sample
deserialize(uint8_t*& buffer)
{
    region_sample value;
    utility::parse_value(value.thread_id, buffer);
    utility::parse_value(value.kind, buffer);
    utility::parse_value(value.operation, buffer);
    utility::parse_value(value.correlation_id_internal, buffer);
    utility::parse_value(value.correlation_id_ancestor, buffer);
    utility::parse_value(value.start_timestamp, buffer);
    utility::parse_value(value.end_timestamp, buffer);
    utility::parse_value(value.call_stack, buffer);
    utility::parse_value(value.args_str, buffer);
    utility::parse_value(value.category, buffer);
    return value;
}

template <>
inline size_t
get_size(const region_sample& value)
{
    return utility::get_size_helper(value.thread_id) +
           utility::get_size_helper(value.kind) +
           utility::get_size_helper(value.operation) +
           utility::get_size_helper(value.correlation_id_internal) +
           utility::get_size_helper(value.correlation_id_ancestor) +
           utility::get_size_helper(value.start_timestamp) +
           utility::get_size_helper(value.end_timestamp) +
           utility::get_size_helper(value.call_stack.c_str()) +
           utility::get_size_helper(value.args_str.c_str()) +
           utility::get_size_helper(value.category.c_str());
}

struct in_time_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::in_time_sample;

    in_time_sample() = default;
    in_time_sample(std::string _track_name, size_t _timestamp_ns,
                   std::string _event_metadata, size_t _stack_id, size_t _parent_stack_id,
                   size_t _correlation_id, std::string _call_stack,
                   std::string _line_info)
    : track_name(std::move(_track_name))
    , timestamp_ns(_timestamp_ns)
    , event_metadata(std::move(_event_metadata))
    , stack_id(_stack_id)
    , parent_stack_id(_parent_stack_id)
    , correlation_id(_correlation_id)
    , call_stack(std::move(_call_stack))
    , line_info(std::move(_line_info))
    {}

    std::string track_name;
    size_t      timestamp_ns;
    std::string event_metadata;
    size_t      stack_id;
    size_t      parent_stack_id;
    size_t      correlation_id;
    std::string call_stack;
    std::string line_info;
};

template <>
inline void
serialize(uint8_t* buffer, const in_time_sample& value)
{
    size_t position = 0;
    utility::store_value(value.track_name.c_str(), buffer, position);
    utility::store_value(value.timestamp_ns, buffer, position);
    utility::store_value(value.event_metadata.c_str(), buffer, position);
    utility::store_value(value.stack_id, buffer, position);
    utility::store_value(value.parent_stack_id, buffer, position);
    utility::store_value(value.correlation_id, buffer, position);
    utility::store_value(value.call_stack.c_str(), buffer, position);
    utility::store_value(value.line_info.c_str(), buffer, position);
}

template <>
inline in_time_sample
deserialize(uint8_t*& buffer)
{
    in_time_sample value;
    utility::parse_value(value.track_name, buffer);
    utility::parse_value(value.timestamp_ns, buffer);
    utility::parse_value(value.event_metadata, buffer);
    utility::parse_value(value.stack_id, buffer);
    utility::parse_value(value.parent_stack_id, buffer);
    utility::parse_value(value.correlation_id, buffer);
    utility::parse_value(value.call_stack, buffer);
    utility::parse_value(value.line_info, buffer);
    return value;
}

template <>
inline size_t
get_size(const in_time_sample& value)
{
    return utility::get_size_helper(value.track_name.c_str()) +
           utility::get_size_helper(value.timestamp_ns) +
           utility::get_size_helper(value.event_metadata.c_str()) +
           utility::get_size_helper(value.stack_id) +
           utility::get_size_helper(value.parent_stack_id) +
           utility::get_size_helper(value.correlation_id) +
           utility::get_size_helper(value.call_stack.c_str()) +
           utility::get_size_helper(value.line_info.c_str());
}

struct pmc_event_with_sample : in_time_sample
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::pmc_event_with_sample;

    pmc_event_with_sample() = default;
    pmc_event_with_sample(std::string _track_name, size_t _timestamp_ns,
                          std::string _event_metadata, size_t _stack_id,
                          size_t _parent_stack_id, size_t _correlation_id,
                          std::string _call_stack, std::string _line_info,
                          uint32_t _device_id, uint8_t _device_type,
                          std::string _pmc_info_name, double _value)
    : in_time_sample(std::move(_track_name), _timestamp_ns, std::move(_event_metadata),
                     _stack_id, _parent_stack_id, _correlation_id, std::move(_call_stack),
                     std::move(_line_info))
    , device_id(_device_id)
    , device_type(_device_type)
    , pmc_info_name(std::move(_pmc_info_name))
    , value(_value)
    {}

    uint32_t    device_id;
    uint8_t     device_type;
    std::string pmc_info_name;
    double      value;
};

template <>
inline void
serialize(uint8_t* buffer, const pmc_event_with_sample& value)
{
    size_t position = 0;
    utility::store_value(value.track_name.c_str(), buffer, position);
    utility::store_value(value.timestamp_ns, buffer, position);
    utility::store_value(value.event_metadata.c_str(), buffer, position);
    utility::store_value(value.stack_id, buffer, position);
    utility::store_value(value.parent_stack_id, buffer, position);
    utility::store_value(value.correlation_id, buffer, position);
    utility::store_value(value.call_stack.c_str(), buffer, position);
    utility::store_value(value.line_info.c_str(), buffer, position);
    utility::store_value(value.device_id, buffer, position);
    utility::store_value(value.device_type, buffer, position);
    utility::store_value(value.pmc_info_name.c_str(), buffer, position);
    utility::store_value(value.value, buffer, position);
}

template <>
inline pmc_event_with_sample
deserialize(uint8_t*& buffer)
{
    pmc_event_with_sample value;
    utility::parse_value(value.track_name, buffer);
    utility::parse_value(value.timestamp_ns, buffer);
    utility::parse_value(value.event_metadata, buffer);
    utility::parse_value(value.stack_id, buffer);
    utility::parse_value(value.parent_stack_id, buffer);
    utility::parse_value(value.correlation_id, buffer);
    utility::parse_value(value.call_stack, buffer);
    utility::parse_value(value.line_info, buffer);
    utility::parse_value(value.device_id, buffer);
    utility::parse_value(value.device_type, buffer);
    utility::parse_value(value.pmc_info_name, buffer);
    utility::parse_value(value.value, buffer);
    return value;
}

template <>
inline size_t
get_size(const pmc_event_with_sample& value)
{
    return utility::get_size_helper(value.track_name.c_str()) +
           utility::get_size_helper(value.timestamp_ns) +
           utility::get_size_helper(value.event_metadata.c_str()) +
           utility::get_size_helper(value.stack_id) +
           utility::get_size_helper(value.parent_stack_id) +
           utility::get_size_helper(value.correlation_id) +
           utility::get_size_helper(value.call_stack.c_str()) +
           utility::get_size_helper(value.line_info.c_str()) +
           utility::get_size_helper(value.device_id) +
           utility::get_size_helper(value.device_type) +
           utility::get_size_helper(value.pmc_info_name.c_str()) +
           utility::get_size_helper(value.value);
}

struct amd_smi_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::amd_smi_sample;

    enum class settings_positions : uint8_t
    {
        busy = 0,
        temp,
        power,
        mem_usage,
        vcn_activity,
        jpeg_activity
    };

    amd_smi_sample() = default;
    amd_smi_sample(uint64_t _settings, uint32_t _device_id, size_t _timestamp,
                   uint32_t _gfx_activity, uint32_t _umc_activity, uint32_t _mm_activity,
                   uint32_t _power, int64_t _temperature, size_t _mem_usage,
                   std::vector<uint8_t> _xcp_activity)
    : settings(_settings)
    , device_id(_device_id)
    , timestamp(_timestamp)
    , gfx_activity(_gfx_activity)
    , umc_activity(_umc_activity)
    , mm_activity(_mm_activity)
    , power(_power)
    , temperature(_temperature)
    , mem_usage(_mem_usage)
    , xcp_activity(std::move(_xcp_activity))
    {}

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

template <>
inline void
serialize(uint8_t* buffer, const amd_smi_sample& value)
{
    size_t position = 0;
    utility::store_value(value.settings, buffer, position);
    utility::store_value(value.device_id, buffer, position);
    utility::store_value(value.timestamp, buffer, position);
    utility::store_value(value.gfx_activity, buffer, position);
    utility::store_value(value.umc_activity, buffer, position);
    utility::store_value(value.mm_activity, buffer, position);
    utility::store_value(value.power, buffer, position);
    utility::store_value(value.temperature, buffer, position);
    utility::store_value(value.mem_usage, buffer, position);
    utility::store_value(value.xcp_activity, buffer, position);
}

template <>
inline amd_smi_sample
deserialize(uint8_t*& buffer)
{
    amd_smi_sample value;
    utility::parse_value(value.settings, buffer);
    utility::parse_value(value.device_id, buffer);
    utility::parse_value(value.timestamp, buffer);
    utility::parse_value(value.gfx_activity, buffer);
    utility::parse_value(value.umc_activity, buffer);
    utility::parse_value(value.mm_activity, buffer);
    utility::parse_value(value.power, buffer);
    utility::parse_value(value.temperature, buffer);
    utility::parse_value(value.mem_usage, buffer);
    utility::parse_value(value.xcp_activity, buffer);
    return value;
}

template <>
inline size_t
get_size(const amd_smi_sample& value)
{
    return utility::get_size_helper(value.settings) +
           utility::get_size_helper(value.device_id) +
           utility::get_size_helper(value.timestamp) +
           utility::get_size_helper(value.gfx_activity) +
           utility::get_size_helper(value.umc_activity) +
           utility::get_size_helper(value.mm_activity) +
           utility::get_size_helper(value.power) +
           utility::get_size_helper(value.temperature) +
           utility::get_size_helper(value.mem_usage) +
           utility::get_size_helper(value.xcp_activity);
}

struct cpu_freq_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::cpu_freq_sample;

    cpu_freq_sample() = default;
    cpu_freq_sample(size_t _timestamp, int64_t _page_rss, int64_t _virt_mem_usage,
                    int64_t _peak_rss, int64_t _context_switch_count,
                    int64_t _page_faults, int64_t _user_mode_time,
                    int64_t _kernel_mode_time, std::vector<uint8_t> _freqs)
    : timestamp(_timestamp)
    , page_rss(_page_rss)
    , virt_mem_usage(_virt_mem_usage)
    , peak_rss(_peak_rss)
    , context_switch_count(_context_switch_count)
    , page_faults(_page_faults)
    , user_mode_time(_user_mode_time)
    , kernel_mode_time(_kernel_mode_time)
    , freqs(std::move(_freqs))
    {}

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

template <>
inline void
serialize(uint8_t* buffer, const cpu_freq_sample& value)
{
    size_t position = 0;
    utility::store_value(value.timestamp, buffer, position);
    utility::store_value(value.page_rss, buffer, position);
    utility::store_value(value.virt_mem_usage, buffer, position);
    utility::store_value(value.peak_rss, buffer, position);
    utility::store_value(value.context_switch_count, buffer, position);
    utility::store_value(value.page_faults, buffer, position);
    utility::store_value(value.user_mode_time, buffer, position);
    utility::store_value(value.kernel_mode_time, buffer, position);
    utility::store_value(value.freqs, buffer, position);
}

template <>
inline cpu_freq_sample
deserialize(uint8_t*& buffer)
{
    cpu_freq_sample value;
    utility::parse_value(value.timestamp, buffer);
    utility::parse_value(value.page_rss, buffer);
    utility::parse_value(value.virt_mem_usage, buffer);
    utility::parse_value(value.peak_rss, buffer);
    utility::parse_value(value.context_switch_count, buffer);
    utility::parse_value(value.page_faults, buffer);
    utility::parse_value(value.user_mode_time, buffer);
    utility::parse_value(value.kernel_mode_time, buffer);
    utility::parse_value(value.freqs, buffer);
    return value;
}

template <>
inline size_t
get_size(const cpu_freq_sample& value)
{
    return utility::get_size_helper(value.timestamp) +
           utility::get_size_helper(value.page_rss) +
           utility::get_size_helper(value.virt_mem_usage) +
           utility::get_size_helper(value.peak_rss) +
           utility::get_size_helper(value.context_switch_count) +
           utility::get_size_helper(value.page_faults) +
           utility::get_size_helper(value.user_mode_time) +
           utility::get_size_helper(value.kernel_mode_time) +
           utility::get_size_helper(value.freqs);
}

struct backtrace_region_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::backtrace_region_sample;

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

template <>
inline void
serialize(uint8_t* buffer, const backtrace_region_sample& value)
{
    size_t position = 0;
    utility::store_value(value.type, buffer, position);
    utility::store_value(value.thread_id, buffer, position);
    utility::store_value(value.track_name.c_str(), buffer, position);
    utility::store_value(value.name.c_str(), buffer, position);
    utility::store_value(value.start_timestamp, buffer, position);
    utility::store_value(value.end_timestamp, buffer, position);
    utility::store_value(value.category.c_str(), buffer, position);
    utility::store_value(value.call_stack.c_str(), buffer, position);
    utility::store_value(value.line_info.c_str(), buffer, position);
    utility::store_value(value.extdata.c_str(), buffer, position);
}

template <>
inline backtrace_region_sample
deserialize(uint8_t*& buffer)
{
    backtrace_region_sample value;
    utility::parse_value(value.type, buffer);
    utility::parse_value(value.thread_id, buffer);
    utility::parse_value(value.track_name, buffer);
    utility::parse_value(value.name, buffer);
    utility::parse_value(value.start_timestamp, buffer);
    utility::parse_value(value.end_timestamp, buffer);
    utility::parse_value(value.category, buffer);
    utility::parse_value(value.call_stack, buffer);
    utility::parse_value(value.line_info, buffer);
    utility::parse_value(value.extdata, buffer);
    return value;
}

template <>
inline size_t
get_size(const backtrace_region_sample& value)
{
    return utility::get_size_helper(value.type) +
           utility::get_size_helper(value.thread_id) +
           utility::get_size_helper(value.track_name.c_str()) +
           utility::get_size_helper(value.name.c_str()) +
           utility::get_size_helper(value.start_timestamp) +
           utility::get_size_helper(value.end_timestamp) +
           utility::get_size_helper(value.category.c_str()) +
           utility::get_size_helper(value.call_stack.c_str()) +
           utility::get_size_helper(value.line_info.c_str()) +
           utility::get_size_helper(value.extdata.c_str());
}

}  // namespace trace_cache
}  // namespace rocprofsys
