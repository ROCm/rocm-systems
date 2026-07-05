// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "core/trace_cache/cacheable.hpp"

#include <cstdint>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{

enum class type_identifier_t : std::uint32_t
{
    in_time_sample          = 0x0000,
    pmc_event_with_sample   = 0x0001,
    region                  = 0x0002,
    kernel_dispatch         = 0x0003,
    memory_copy             = 0x0004,
    memory_alloc            = 0x0005,
    gpu_pmc_sample          = 0x0006,
    cpu_pmc_sample          = 0x0007,
    backtrace_region_sample = 0x0008,
    scratch_memory          = 0x0009,
    ainic_pmc_sample        = 0x000A,
    kfd_sample              = 0x000B,
    gpu_perf_counter_sample = 0x000C,
    fragmented_space        = 0xFFFF
};

struct kernel_dispatch_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::kernel_dispatch;

    kernel_dispatch_sample() = default;
    kernel_dispatch_sample(
        std::uint64_t _start_timestamp, std::uint64_t _end_timestamp,
        std::uint64_t _thread_id, std::uint64_t _agent_id_handle,
        std::uint64_t _kernel_id, std::uint64_t _dispatch_id,
        std::uint64_t _queue_id_handle, std::uint64_t _correlation_id_internal,
        std::uint64_t _correlation_id_ancestor, std::uint32_t _private_segment_size,
        std::uint32_t _group_segment_size, std::uint32_t _workgroup_size_x,
        std::uint32_t _workgroup_size_y, std::uint32_t _workgroup_size_z,
        std::uint32_t _grid_size_x, std::uint32_t _grid_size_y,
        std::uint32_t _grid_size_z, size_t _stream_handle)
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

    std::uint64_t start_timestamp;
    std::uint64_t end_timestamp;
    std::uint64_t thread_id;
    std::uint64_t agent_id_handle;
    std::uint64_t kernel_id;
    std::uint64_t dispatch_id;
    std::uint64_t queue_id_handle;
    std::uint64_t correlation_id_internal;
    std::uint64_t correlation_id_ancestor;
    std::uint32_t private_segment_size;
    std::uint32_t group_segment_size;
    std::uint32_t workgroup_size_x;
    std::uint32_t workgroup_size_y;
    std::uint32_t workgroup_size_z;
    std::uint32_t grid_size_x;
    std::uint32_t grid_size_y;
    std::uint32_t grid_size_z;
    size_t        stream_handle;
};

template <>
inline void
serialize(std::uint8_t* buffer, const kernel_dispatch_sample& item)
{
    utility::store_value(
        buffer, item.start_timestamp, item.end_timestamp, item.thread_id,
        item.agent_id_handle, item.kernel_id, item.dispatch_id, item.queue_id_handle,
        item.correlation_id_internal, item.correlation_id_ancestor,
        item.private_segment_size, item.group_segment_size, item.workgroup_size_x,
        item.workgroup_size_y, item.workgroup_size_z, item.grid_size_x, item.grid_size_y,
        item.grid_size_z, static_cast<std::uint64_t>(item.stream_handle));
}

template <>
inline kernel_dispatch_sample
deserialize(std::uint8_t*& buffer)
{
    kernel_dispatch_sample item;
    std::uint64_t          stream_handle;
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
        item.grid_size_z, static_cast<std::uint64_t>(item.stream_handle));
}

struct scratch_memory_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::scratch_memory;

    scratch_memory_sample() = default;
    scratch_memory_sample(std::uint64_t _start_timestamp, std::uint64_t _end_timestamp,
                          std::uint64_t _thread_id, std::uint64_t _agent_id_handle,
                          std::uint64_t _queue_id_handle, std::int32_t _kind,
                          std::int32_t _operation, std::int32_t _flags,
                          std::uint64_t _allocation_size,
                          std::uint64_t _correlation_id_internal,
                          std::uint64_t _correlation_id_ancestor, size_t _stream_handle)
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

    std::uint64_t start_timestamp;
    std::uint64_t end_timestamp;
    std::uint64_t thread_id;
    std::uint64_t agent_id_handle;
    std::uint64_t queue_id_handle;
    std::int32_t  kind;
    std::int32_t  operation;
    std::int32_t  flags;
    std::uint64_t allocation_size;
    std::uint64_t correlation_id_internal;
    std::uint64_t correlation_id_ancestor;
    size_t        stream_handle;
};

template <>
inline void
serialize(std::uint8_t* buffer, const scratch_memory_sample& item)
{
    utility::store_value(buffer, item.start_timestamp, item.end_timestamp, item.thread_id,
                         item.agent_id_handle, item.queue_id_handle, item.kind,
                         item.operation, item.flags, item.allocation_size,
                         item.correlation_id_internal, item.correlation_id_ancestor,
                         static_cast<std::uint64_t>(item.stream_handle));
}

template <>
inline scratch_memory_sample
deserialize(std::uint8_t*& buffer)
{
    scratch_memory_sample item;
    std::uint64_t         stream_handle;
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
                             static_cast<std::uint64_t>(item.stream_handle));
}

struct memory_copy_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::memory_copy;

    memory_copy_sample() = default;
    memory_copy_sample(std::uint64_t _start_timestamp, std::uint64_t _end_timestamp,
                       std::uint64_t _thread_id, std::uint64_t _dst_agent_id_handle,
                       std::uint64_t _src_agent_id_handle, std::int32_t _kind,
                       std::int32_t _operation, std::uint64_t _bytes,
                       std::uint64_t _correlation_id_internal,
                       std::uint64_t _correlation_id_ancestor,
                       std::uint64_t _dst_address_value, std::uint64_t _src_address_value,
                       size_t _stream_handle)
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

    std::uint64_t start_timestamp;
    std::uint64_t end_timestamp;
    std::uint64_t thread_id;
    std::uint64_t dst_agent_id_handle;
    std::uint64_t src_agent_id_handle;
    std::int32_t  kind;
    std::int32_t  operation;
    std::uint64_t bytes;
    std::uint64_t correlation_id_internal;
    std::uint64_t correlation_id_ancestor;
    std::uint64_t dst_address_value;
    std::uint64_t src_address_value;
    size_t        stream_handle;
};

template <>
inline void
serialize(std::uint8_t* buffer, const memory_copy_sample& item)
{
    utility::store_value(buffer, item.start_timestamp, item.end_timestamp, item.thread_id,
                         item.dst_agent_id_handle, item.src_agent_id_handle, item.kind,
                         item.operation, item.bytes, item.correlation_id_internal,
                         item.correlation_id_ancestor, item.dst_address_value,
                         item.src_address_value,
                         static_cast<std::uint64_t>(item.stream_handle));
}

template <>
inline memory_copy_sample
deserialize(std::uint8_t*& buffer)
{
    memory_copy_sample item;
    std::uint64_t      stream_handle;
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
                             static_cast<std::uint64_t>(item.stream_handle));
}

struct memory_allocate_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::memory_alloc;

    memory_allocate_sample() = default;
    memory_allocate_sample(std::uint64_t _start_timestamp, std::uint64_t _end_timestamp,
                           std::uint64_t _thread_id, std::uint64_t _agent_id_handle,
                           std::int32_t _kind, std::int32_t _operation,
                           std::uint64_t _allocation_size,
                           std::uint64_t _correlation_id_internal,
                           std::uint64_t _correlation_id_ancestor,
                           std::uint64_t _address_value, size_t _stream_handle)
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

    std::uint64_t start_timestamp;
    std::uint64_t end_timestamp;
    std::uint64_t thread_id;
    std::uint64_t agent_id_handle;
    std::int32_t  kind;
    std::int32_t  operation;
    std::uint64_t allocation_size;
    std::uint64_t correlation_id_internal;
    std::uint64_t correlation_id_ancestor;
    std::uint64_t address_value;
    size_t        stream_handle;
};

template <>
inline void
serialize(std::uint8_t* buffer, const memory_allocate_sample& item)
{
    utility::store_value(buffer, item.start_timestamp, item.end_timestamp, item.thread_id,
                         item.agent_id_handle, item.kind, item.operation,
                         item.allocation_size, item.correlation_id_internal,
                         item.correlation_id_ancestor, item.address_value,
                         static_cast<std::uint64_t>(item.stream_handle));
}

template <>
inline memory_allocate_sample
deserialize(std::uint8_t*& buffer)
{
    memory_allocate_sample item;
    std::uint64_t          stream_handle;
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
                             static_cast<std::uint64_t>(item.stream_handle));
}

struct region_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::region;

    region_sample() = default;
    region_sample(std::uint64_t _thread_id, std::string _name,
                  std::uint64_t _correlation_id_internal,
                  std::uint64_t _correlation_id_ancestor, std::uint64_t _start_timestamp,
                  std::uint64_t _end_timestamp, std::string _call_stack,
                  std::string _args_str, std::string _category)
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

    std::uint64_t thread_id;
    std::string   name;
    std::uint64_t correlation_id_internal;
    std::uint64_t correlation_id_ancestor;
    std::uint64_t start_timestamp;
    std::uint64_t end_timestamp;
    std::string   call_stack;
    std::string   args_str;
    std::string   category;
};

template <>
inline void
serialize(std::uint8_t* buffer, const region_sample& item)
{
    utility::store_value(
        buffer, item.thread_id, std::string_view(item.name), item.correlation_id_internal,
        item.correlation_id_ancestor, item.start_timestamp, item.end_timestamp,
        std::string_view(item.call_stack), std::string_view(item.args_str),
        std::string_view(item.category));
}

template <>
inline region_sample
deserialize(std::uint8_t*& buffer)
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
serialize(std::uint8_t* buffer, const in_time_sample& item)
{
    utility::store_value(
        buffer, item.category_enum_id, std::string_view(item.track_name),
        static_cast<std::uint64_t>(item.timestamp_ns),
        std::string_view(item.event_metadata), static_cast<std::uint64_t>(item.stack_id),
        static_cast<std::uint64_t>(item.parent_stack_id),
        static_cast<std::uint64_t>(item.correlation_id),
        std::string_view(item.call_stack), std::string_view(item.line_info));
}

template <>
inline in_time_sample
deserialize(std::uint8_t*& buffer)
{
    in_time_sample   item;
    size_t           category_enum_id;
    std::string_view track_name_view, event_metadata_view, call_stack_view,
        line_info_view;
    std::uint64_t timestamp_ns, stack_id, parent_stack_id, correlation_id;
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
        static_cast<std::uint64_t>(item.timestamp_ns),
        std::string_view(item.event_metadata), static_cast<std::uint64_t>(item.stack_id),
        static_cast<std::uint64_t>(item.parent_stack_id),
        static_cast<std::uint64_t>(item.correlation_id),
        std::string_view(item.call_stack), std::string_view(item.line_info));
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
                          std::string _line_info, std::uint32_t _device_id,
                          std::uint8_t _device_type, std::string _pmc_info_name,
                          double _value, std::optional<std::int64_t> _system_tid)
    : in_time_sample(_category_enum_id, std::move(_track_name), _timestamp_ns,
                     std::move(_event_metadata), _stack_id, _parent_stack_id,
                     _correlation_id, std::move(_call_stack), std::move(_line_info))
    , device_id(_device_id)
    , device_type(_device_type)
    , pmc_info_name(std::move(_pmc_info_name))
    , value(_value)
    , system_tid(_system_tid)
    {}

    std::uint32_t               device_id;
    std::uint8_t                device_type;
    std::string                 pmc_info_name;
    double                      value;
    std::optional<std::int64_t> system_tid;
};

template <>
inline void
serialize(std::uint8_t* buffer, const pmc_event_with_sample& item)
{
    utility::store_value(
        buffer, item.category_enum_id, std::string_view(item.track_name),
        static_cast<std::uint64_t>(item.timestamp_ns),
        std::string_view(item.event_metadata), static_cast<std::uint64_t>(item.stack_id),
        static_cast<std::uint64_t>(item.parent_stack_id),
        static_cast<std::uint64_t>(item.correlation_id),
        std::string_view(item.call_stack), std::string_view(item.line_info),
        item.device_id, item.device_type, std::string_view(item.pmc_info_name),
        item.value, item.system_tid);
}

template <>
inline pmc_event_with_sample
deserialize(std::uint8_t*& buffer)
{
    pmc_event_with_sample item;
    size_t                category_enum_id;
    std::string_view      track_name_view, event_metadata_view, call_stack_view,
        line_info_view, pmc_info_name_view;
    std::uint64_t timestamp_ns, stack_id, parent_stack_id, correlation_id;
    utility::parse_value(buffer, category_enum_id, track_name_view, timestamp_ns,
                         event_metadata_view, stack_id, parent_stack_id, correlation_id,
                         call_stack_view, line_info_view, item.device_id,
                         item.device_type, pmc_info_name_view, item.value,
                         item.system_tid);
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
        static_cast<std::uint64_t>(item.timestamp_ns),
        std::string_view(item.event_metadata), static_cast<std::uint64_t>(item.stack_id),
        static_cast<std::uint64_t>(item.parent_stack_id),
        static_cast<std::uint64_t>(item.correlation_id),
        std::string_view(item.call_stack), std::string_view(item.line_info),
        item.device_id, item.device_type, std::string_view(item.pmc_info_name),
        item.value, item.system_tid);
}

struct backtrace_region_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::backtrace_region_sample;

    backtrace_region_sample() = default;
    backtrace_region_sample(std::uint32_t _type, std::uint64_t _thread_id,
                            std::string _track_name, std::string _name,
                            std::uint64_t _start_timestamp, std::uint64_t _end_timestamp,
                            std::string _category, std::string _call_stack,
                            std::string _line_info, std::string _extdata)
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

    std::uint32_t type;
    std::uint64_t thread_id;
    std::string   track_name;
    std::string   name;
    std::uint64_t start_timestamp;
    std::uint64_t end_timestamp;
    std::string   category;
    std::string   call_stack;
    std::string   line_info;
    std::string   extdata;
};

template <>
inline void
serialize(std::uint8_t* buffer, const backtrace_region_sample& item)
{
    utility::store_value(
        buffer, item.type, item.thread_id, std::string_view(item.track_name),
        std::string_view(item.name), item.start_timestamp, item.end_timestamp,
        std::string_view(item.category), std::string_view(item.call_stack),
        std::string_view(item.line_info), std::string_view(item.extdata));
}

template <>
inline backtrace_region_sample
deserialize(std::uint8_t*& buffer)
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

struct kfd_sample : cacheable_t
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::kfd_sample;

    kfd_sample() = default;
    kfd_sample(std::uint64_t _thread_id, std::string _name,
               std::uint64_t _start_timestamp, std::uint64_t _end_timestamp,
               std::string _args_str, std::string _category, std::string _track_name,
               std::string _event_metadata, std::uint32_t _device_id,
               std::uint8_t _device_type, std::string _pmc_info_name, double _value,
               std::optional<std::int64_t> _system_tid)
    : thread_id(_thread_id)
    , name(std::move(_name))
    , start_timestamp(_start_timestamp)
    , end_timestamp(_end_timestamp)
    , args_str(std::move(_args_str))
    , category(std::move(_category))
    , track_name(std::move(_track_name))
    , event_metadata(std::move(_event_metadata))
    , device_id(_device_id)
    , device_type(_device_type)
    , pmc_info_name(std::move(_pmc_info_name))
    , value(_value)
    , system_tid(_system_tid)
    {}

    std::uint64_t               thread_id;
    std::string                 name;
    std::uint64_t               start_timestamp;
    std::uint64_t               end_timestamp;
    std::string                 args_str;
    std::string                 category;
    std::string                 track_name;
    std::string                 event_metadata;
    std::uint32_t               device_id;
    std::uint8_t                device_type;
    std::string                 pmc_info_name;
    double                      value;
    std::optional<std::int64_t> system_tid;
};

template <>
inline void
serialize(std::uint8_t* buffer, const kfd_sample& item)
{
    utility::store_value(
        buffer, item.thread_id, std::string_view(item.name), item.start_timestamp,
        item.end_timestamp, std::string_view(item.args_str),
        std::string_view(item.category), std::string_view(item.track_name),
        std::string_view(item.event_metadata), item.device_id, item.device_type,
        std::string_view(item.pmc_info_name), item.value, item.system_tid);
}

template <>
inline kfd_sample
deserialize(std::uint8_t*& buffer)
{
    kfd_sample       item;
    std::string_view name_view, args_str_view, category_view, track_name_view,
        event_metadata_view, pmc_info_name_view;
    utility::parse_value(buffer, item.thread_id, name_view, item.start_timestamp,
                         item.end_timestamp, args_str_view, category_view,
                         track_name_view, event_metadata_view, item.device_id,
                         item.device_type, pmc_info_name_view, item.value,
                         item.system_tid);
    item.name           = std::string(name_view);
    item.args_str       = std::string(args_str_view);
    item.category       = std::string(category_view);
    item.track_name     = std::string(track_name_view);
    item.event_metadata = std::string(event_metadata_view);
    item.pmc_info_name  = std::string(pmc_info_name_view);
    return item;
}

template <>
inline size_t
get_size(const kfd_sample& item)
{
    return utility::get_size(
        item.thread_id, std::string_view(item.name), item.start_timestamp,
        item.end_timestamp, std::string_view(item.args_str),
        std::string_view(item.category), std::string_view(item.track_name),
        std::string_view(item.event_metadata), item.device_id, item.device_type,
        std::string_view(item.pmc_info_name), item.value, item.system_tid);
}

}  // namespace trace_cache
}  // namespace rocprofsys
