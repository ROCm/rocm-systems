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

#include <cstdint>
#include <stdint.h>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{

enum class type_identifier_t : uint32_t
{
    in_time_sample          = 0x0000,
    pmc_event_with_sample   = 0x0001,
    region                  = 0x0002,
    kernel_dispatch         = 0x0003,
    memory_copy             = 0x0004,
    memory_alloc            = 0x0005,
    amd_smi_sample          = 0x0006,
    cpu_freq_sample         = 0x0007,
    backtrace_region_sample = 0x0008,
    scratch_memory          = 0x0009,
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
serialize(uint8_t* buffer, const kernel_dispatch_sample& item)
{
    utility::store_value(
        buffer, item.start_timestamp, item.end_timestamp, item.thread_id,
        item.agent_id_handle, item.kernel_id, item.dispatch_id, item.queue_id_handle,
        item.correlation_id_internal, item.correlation_id_ancestor,
        item.private_segment_size, item.group_segment_size, item.workgroup_size_x,
        item.workgroup_size_y, item.workgroup_size_z, item.grid_size_x, item.grid_size_y,
        item.grid_size_z, static_cast<uint64_t>(item.stream_handle));
}

template <>
inline kernel_dispatch_sample
deserialize(uint8_t*& buffer)
{
    kernel_dispatch_sample item;
    uint64_t               stream_handle;
    utility::parse_value(buffer, item.start_timestamp, item.end_timestamp, item.thread_id,
                         item.agent_id_handle, item.kernel_id, item.dispatch_id,
                         item.queue_id_handle, item.correlation_id_internal,
                         item.correlation_id_ancestor, item.private_segment_size,
                         item.group_segment_size, item.workgroup_size_x,
                         item.workgroup_size_y, item.workgroup_size_z, item.grid_size_x,
                         item.grid_size_y, item.grid_size_z, stream_handle);
    item.stream_handle = stream_handle;
    return item;
}

template <>
inline size_t
get_size(const kernel_dispatch_sample& item)
{
    return utility::get_size(
        item.start_timestamp, item.end_timestamp, item.thread_id, item.agent_id_handle,
        item.kernel_id, item.dispatch_id, item.queue_id_handle,
        item.correlation_id_internal, item.correlation_id_ancestor,
        item.private_segment_size, item.group_segment_size, item.workgroup_size_x,
        item.workgroup_size_y, item.workgroup_size_z, item.grid_size_x, item.grid_size_y,
        item.grid_size_z, static_cast<uint64_t>(item.stream_handle));
}

struct scratch_memory_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::scratch_memory;

    scratch_memory_sample() = default;
    scratch_memory_sample(uint64_t _start_timestamp, uint64_t _end_timestamp,
                          uint64_t _thread_id, uint64_t _agent_id_handle,
                          uint64_t _queue_id_handle, int32_t _kind, int32_t _operation,
                          int32_t _flags, uint64_t _allocation_size,
                          uint64_t _correlation_id_internal,
                          uint64_t _correlation_id_ancestor, size_t _stream_handle)
    : start_timestamp(_start_timestamp)
    , end_timestamp(_end_timestamp)
    , thread_id(_thread_id)
    , agent_id_handle(_agent_id_handle)
    , queue_id_handle(_queue_id_handle)
    , kind(_kind)
    , operation(_operation)
    , flags(_flags)
    , allocation_size(_allocation_size)
    , correlation_id_internal(_correlation_id_internal)
    , correlation_id_ancestor(_correlation_id_ancestor)
    , stream_handle(_stream_handle)
    {}

    uint64_t start_timestamp;
    uint64_t end_timestamp;
    uint64_t thread_id;
    uint64_t agent_id_handle;
    uint64_t queue_id_handle;
    int32_t  kind;
    int32_t  operation;
    int32_t  flags;
    uint64_t allocation_size;
    uint64_t correlation_id_internal;
    uint64_t correlation_id_ancestor;
    size_t   stream_handle;
};

template <>
inline void
serialize(uint8_t* buffer, const scratch_memory_sample& item)
{
    utility::store_value(buffer, item.start_timestamp, item.end_timestamp, item.thread_id,
                         item.agent_id_handle, item.queue_id_handle, item.kind,
                         item.operation, item.flags, item.allocation_size,
                         item.correlation_id_internal, item.correlation_id_ancestor,
                         static_cast<uint64_t>(item.stream_handle));
}

template <>
inline scratch_memory_sample
deserialize(uint8_t*& buffer)
{
    scratch_memory_sample item;
    uint64_t              stream_handle;
    utility::parse_value(buffer, item.start_timestamp, item.end_timestamp, item.thread_id,
                         item.agent_id_handle, item.queue_id_handle, item.kind,
                         item.operation, item.flags, item.allocation_size,
                         item.correlation_id_internal, item.correlation_id_ancestor,
                         stream_handle);
    item.stream_handle = stream_handle;
    return item;
}

template <>
inline size_t
get_size(const scratch_memory_sample& item)
{
    return utility::get_size(item.start_timestamp, item.end_timestamp, item.thread_id,
                             item.agent_id_handle, item.queue_id_handle, item.kind,
                             item.operation, item.flags, item.allocation_size,
                             item.correlation_id_internal, item.correlation_id_ancestor,
                             static_cast<uint64_t>(item.stream_handle));
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
serialize(uint8_t* buffer, const memory_copy_sample& item)
{
    utility::store_value(buffer, item.start_timestamp, item.end_timestamp, item.thread_id,
                         item.dst_agent_id_handle, item.src_agent_id_handle, item.kind,
                         item.operation, item.bytes, item.correlation_id_internal,
                         item.correlation_id_ancestor, item.dst_address_value,
                         item.src_address_value,
                         static_cast<uint64_t>(item.stream_handle));
}

template <>
inline memory_copy_sample
deserialize(uint8_t*& buffer)
{
    memory_copy_sample item;
    uint64_t           stream_handle;
    utility::parse_value(buffer, item.start_timestamp, item.end_timestamp, item.thread_id,
                         item.dst_agent_id_handle, item.src_agent_id_handle, item.kind,
                         item.operation, item.bytes, item.correlation_id_internal,
                         item.correlation_id_ancestor, item.dst_address_value,
                         item.src_address_value, stream_handle);
    item.stream_handle = stream_handle;
    return item;
}

template <>
inline size_t
get_size(const memory_copy_sample& item)
{
    return utility::get_size(item.start_timestamp, item.end_timestamp, item.thread_id,
                             item.dst_agent_id_handle, item.src_agent_id_handle,
                             item.kind, item.operation, item.bytes,
                             item.correlation_id_internal, item.correlation_id_ancestor,
                             item.dst_address_value, item.src_address_value,
                             static_cast<uint64_t>(item.stream_handle));
}

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
serialize(uint8_t* buffer, const memory_allocate_sample& item)
{
    utility::store_value(buffer, item.start_timestamp, item.end_timestamp, item.thread_id,
                         item.agent_id_handle, item.kind, item.operation,
                         item.allocation_size, item.correlation_id_internal,
                         item.correlation_id_ancestor, item.address_value,
                         static_cast<uint64_t>(item.stream_handle));
}

template <>
inline memory_allocate_sample
deserialize(uint8_t*& buffer)
{
    memory_allocate_sample item;
    uint64_t               stream_handle;
    utility::parse_value(buffer, item.start_timestamp, item.end_timestamp, item.thread_id,
                         item.agent_id_handle, item.kind, item.operation,
                         item.allocation_size, item.correlation_id_internal,
                         item.correlation_id_ancestor, item.address_value, stream_handle);
    item.stream_handle = stream_handle;
    return item;
}

template <>
inline size_t
get_size(const memory_allocate_sample& item)
{
    return utility::get_size(item.start_timestamp, item.end_timestamp, item.thread_id,
                             item.agent_id_handle, item.kind, item.operation,
                             item.allocation_size, item.correlation_id_internal,
                             item.correlation_id_ancestor, item.address_value,
                             static_cast<uint64_t>(item.stream_handle));
}

struct region_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::region;

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
    uint64_t    correlation_id_internal;
    uint64_t    correlation_id_ancestor;
    uint64_t    start_timestamp;
    uint64_t    end_timestamp;
    std::string call_stack;
    std::string args_str;
    std::string category;
};

template <>
inline void
serialize(uint8_t* buffer, const region_sample& item)
{
    utility::store_value(
        buffer, item.thread_id, std::string_view(item.name), item.correlation_id_internal,
        item.correlation_id_ancestor, item.start_timestamp, item.end_timestamp,
        std::string_view(item.call_stack), std::string_view(item.args_str),
        std::string_view(item.category));
}

template <>
inline region_sample
deserialize(uint8_t*& buffer)
{
    region_sample    item;
    std::string_view name_view, call_stack_view, args_str_view, category_view;
    utility::parse_value(buffer, item.thread_id, name_view, item.correlation_id_internal,
                         item.correlation_id_ancestor, item.start_timestamp,
                         item.end_timestamp, call_stack_view, args_str_view,
                         category_view);
    item.name       = std::string(name_view);
    item.call_stack = std::string(call_stack_view);
    item.args_str   = std::string(args_str_view);
    item.category   = std::string(category_view);
    return item;
}

template <>
inline size_t
get_size(const region_sample& item)
{
    return utility::get_size(
        item.thread_id, std::string_view(item.name), item.correlation_id_internal,
        item.correlation_id_ancestor, item.start_timestamp, item.end_timestamp,
        std::string_view(item.call_stack), std::string_view(item.args_str),
        std::string_view(item.category));
}

struct in_time_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::in_time_sample;

    in_time_sample() = default;
    in_time_sample(size_t _category_enum_id, std::string _track_name,
                   size_t _timestamp_ns, std::string _event_metadata, size_t _stack_id,
                   size_t _parent_stack_id, size_t _correlation_id,
                   std::string _call_stack, std::string _line_info)
    : category_enum_id(_category_enum_id)
    , track_name(std::move(_track_name))
    , timestamp_ns(_timestamp_ns)
    , event_metadata(std::move(_event_metadata))
    , stack_id(_stack_id)
    , parent_stack_id(_parent_stack_id)
    , correlation_id(_correlation_id)
    , call_stack(std::move(_call_stack))
    , line_info(std::move(_line_info))
    {}

    size_t      category_enum_id;
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
serialize(uint8_t* buffer, const in_time_sample& item)
{
    utility::store_value(
        buffer, item.category_enum_id, std::string_view(item.track_name),
        static_cast<uint64_t>(item.timestamp_ns), std::string_view(item.event_metadata),
        static_cast<uint64_t>(item.stack_id), static_cast<uint64_t>(item.parent_stack_id),
        static_cast<uint64_t>(item.correlation_id), std::string_view(item.call_stack),
        std::string_view(item.line_info));
}

template <>
inline in_time_sample
deserialize(uint8_t*& buffer)
{
    in_time_sample   item;
    size_t           category_enum_id;
    std::string_view track_name_view, event_metadata_view, call_stack_view,
        line_info_view;
    uint64_t timestamp_ns, stack_id, parent_stack_id, correlation_id;
    utility::parse_value(buffer, category_enum_id, track_name_view, timestamp_ns,
                         event_metadata_view, stack_id, parent_stack_id, correlation_id,
                         call_stack_view, line_info_view);
    item.category_enum_id = category_enum_id;
    item.track_name       = std::string(track_name_view);
    item.timestamp_ns     = timestamp_ns;
    item.event_metadata   = std::string(event_metadata_view);
    item.stack_id         = stack_id;
    item.parent_stack_id  = parent_stack_id;
    item.correlation_id   = correlation_id;
    item.call_stack       = std::string(call_stack_view);
    item.line_info        = std::string(line_info_view);
    return item;
}

template <>
inline size_t
get_size(const in_time_sample& item)
{
    return utility::get_size(
        item.category_enum_id, std::string_view(item.track_name),
        static_cast<uint64_t>(item.timestamp_ns), std::string_view(item.event_metadata),
        static_cast<uint64_t>(item.stack_id), static_cast<uint64_t>(item.parent_stack_id),
        static_cast<uint64_t>(item.correlation_id), std::string_view(item.call_stack),
        std::string_view(item.line_info));
}

struct pmc_event_with_sample : in_time_sample
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::pmc_event_with_sample;

    pmc_event_with_sample() = default;
    pmc_event_with_sample(size_t _category_enum_id, std::string _track_name,
                          size_t _timestamp_ns, std::string _event_metadata,
                          size_t _stack_id, size_t _parent_stack_id,
                          size_t _correlation_id, std::string _call_stack,
                          std::string _line_info, uint32_t _device_id,
                          uint8_t _device_type, std::string _pmc_info_name, double _value)
    : in_time_sample(_category_enum_id, std::move(_track_name), _timestamp_ns,
                     std::move(_event_metadata), _stack_id, _parent_stack_id,
                     _correlation_id, std::move(_call_stack), std::move(_line_info))
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
serialize(uint8_t* buffer, const pmc_event_with_sample& item)
{
    utility::store_value(
        buffer, item.category_enum_id, std::string_view(item.track_name),
        static_cast<uint64_t>(item.timestamp_ns), std::string_view(item.event_metadata),
        static_cast<uint64_t>(item.stack_id), static_cast<uint64_t>(item.parent_stack_id),
        static_cast<uint64_t>(item.correlation_id), std::string_view(item.call_stack),
        std::string_view(item.line_info), item.device_id, item.device_type,
        std::string_view(item.pmc_info_name), item.value);
}

template <>
inline pmc_event_with_sample
deserialize(uint8_t*& buffer)
{
    pmc_event_with_sample item;
    size_t                category_enum_id;
    std::string_view      track_name_view, event_metadata_view, call_stack_view,
        line_info_view, pmc_info_name_view;
    uint64_t timestamp_ns, stack_id, parent_stack_id, correlation_id;
    utility::parse_value(buffer, category_enum_id, track_name_view, timestamp_ns,
                         event_metadata_view, stack_id, parent_stack_id, correlation_id,
                         call_stack_view, line_info_view, item.device_id,
                         item.device_type, pmc_info_name_view, item.value);
    item.category_enum_id = category_enum_id;
    item.track_name       = std::string(track_name_view);
    item.timestamp_ns     = timestamp_ns;
    item.event_metadata   = std::string(event_metadata_view);
    item.stack_id         = stack_id;
    item.parent_stack_id  = parent_stack_id;
    item.correlation_id   = correlation_id;
    item.call_stack       = std::string(call_stack_view);
    item.line_info        = std::string(line_info_view);
    item.pmc_info_name    = std::string(pmc_info_name_view);
    return item;
}

template <>
inline size_t
get_size(const pmc_event_with_sample& item)
{
    return utility::get_size(
        item.category_enum_id, std::string_view(item.track_name),
        static_cast<uint64_t>(item.timestamp_ns), std::string_view(item.event_metadata),
        static_cast<uint64_t>(item.stack_id), static_cast<uint64_t>(item.parent_stack_id),
        static_cast<uint64_t>(item.correlation_id), std::string_view(item.call_stack),
        std::string_view(item.line_info), item.device_id, item.device_type,
        std::string_view(item.pmc_info_name), item.value);
}

struct amd_smi_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::amd_smi_sample;

    amd_smi_sample() = default;
    amd_smi_sample(uint64_t _settings, uint32_t _device_id, size_t _timestamp,
                   uint32_t _gfx_activity, uint32_t _umc_activity, uint32_t _mm_activity,
                   uint32_t _power, int64_t _temperature, size_t _mem_usage,
                   std::vector<uint8_t> _gpu_activity)
    : settings(_settings)
    , device_id(_device_id)
    , timestamp(_timestamp)
    , gfx_activity(_gfx_activity)
    , umc_activity(_umc_activity)
    , mm_activity(_mm_activity)
    , power(_power)
    , temperature(_temperature)
    , mem_usage(_mem_usage)
    , gpu_activity(std::move(_gpu_activity))
    {}

    enum class settings_positions : uint8_t
    {
        busy = 0,
        temp,
        power,
        mem_usage,
        vcn_activity,
        jpeg_activity,
        xgmi,
        pcie
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
    std::vector<uint8_t> gpu_activity;
};

template <>
inline void
serialize(uint8_t* buffer, const amd_smi_sample& item)
{
    utility::store_value(
        buffer, item.settings, item.device_id, static_cast<uint64_t>(item.timestamp),
        item.gfx_activity, item.umc_activity, item.mm_activity, item.power,
        item.temperature, static_cast<uint64_t>(item.mem_usage), item.gpu_activity);
}

template <>
inline amd_smi_sample
deserialize(uint8_t*& buffer)
{
    amd_smi_sample item;
    uint64_t       timestamp, mem_usage;
    utility::parse_value(buffer, item.settings, item.device_id, timestamp,
                         item.gfx_activity, item.umc_activity, item.mm_activity,
                         item.power, item.temperature, mem_usage, item.gpu_activity);
    item.timestamp = timestamp;
    item.mem_usage = mem_usage;
    return item;
}

template <>
inline size_t
get_size(const amd_smi_sample& item)
{
    return utility::get_size(
        item.settings, item.device_id, static_cast<uint64_t>(item.timestamp),
        item.gfx_activity, item.umc_activity, item.mm_activity, item.power,
        item.temperature, static_cast<uint64_t>(item.mem_usage), item.gpu_activity);
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
serialize(uint8_t* buffer, const cpu_freq_sample& item)
{
    utility::store_value(buffer, static_cast<uint64_t>(item.timestamp), item.page_rss,
                         item.virt_mem_usage, item.peak_rss, item.context_switch_count,
                         item.page_faults, item.user_mode_time, item.kernel_mode_time,
                         item.freqs);
}

template <>
inline cpu_freq_sample
deserialize(uint8_t*& buffer)
{
    cpu_freq_sample item;
    uint64_t        timestamp;
    utility::parse_value(buffer, timestamp, item.page_rss, item.virt_mem_usage,
                         item.peak_rss, item.context_switch_count, item.page_faults,
                         item.user_mode_time, item.kernel_mode_time, item.freqs);
    item.timestamp = timestamp;
    return item;
}

template <>
inline size_t
get_size(const cpu_freq_sample& item)
{
    return utility::get_size(static_cast<uint64_t>(item.timestamp), item.page_rss,
                             item.virt_mem_usage, item.peak_rss,
                             item.context_switch_count, item.page_faults,
                             item.user_mode_time, item.kernel_mode_time, item.freqs);
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
    uint64_t    start_timestamp;
    uint64_t    end_timestamp;
    std::string category;
    std::string call_stack;
    std::string line_info;
    std::string extdata;
};

template <>
inline void
serialize(uint8_t* buffer, const backtrace_region_sample& item)
{
    utility::store_value(
        buffer, item.type, item.thread_id, std::string_view(item.track_name),
        std::string_view(item.name), item.start_timestamp, item.end_timestamp,
        std::string_view(item.category), std::string_view(item.call_stack),
        std::string_view(item.line_info), std::string_view(item.extdata));
}

template <>
inline backtrace_region_sample
deserialize(uint8_t*& buffer)
{
    backtrace_region_sample item;
    std::string_view        track_name_view, name_view, category_view, call_stack_view,
        line_info_view, extdata_view;
    utility::parse_value(buffer, item.type, item.thread_id, track_name_view, name_view,
                         item.start_timestamp, item.end_timestamp, category_view,
                         call_stack_view, line_info_view, extdata_view);
    item.track_name = std::string(track_name_view);
    item.name       = std::string(name_view);
    item.category   = std::string(category_view);
    item.call_stack = std::string(call_stack_view);
    item.line_info  = std::string(line_info_view);
    item.extdata    = std::string(extdata_view);
    return item;
}

template <>
inline size_t
get_size(const backtrace_region_sample& item)
{
    return utility::get_size(
        item.type, item.thread_id, std::string_view(item.track_name),
        std::string_view(item.name), item.start_timestamp, item.end_timestamp,
        std::string_view(item.category), std::string_view(item.call_stack),
        std::string_view(item.line_info), std::string_view(item.extdata));
}

}  // namespace trace_cache
}  // namespace rocprofsys
