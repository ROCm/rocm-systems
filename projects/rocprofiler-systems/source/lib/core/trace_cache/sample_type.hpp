// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "core/trace_cache/cacheable.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocprofsys::trace_cache
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
    hipfile_pmc_sample      = 0x000D,
    fragmented_space        = 0xFFFF
};

struct kernel_dispatch_sample
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

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(start_timestamp, end_timestamp, thread_id, agent_id_handle, kernel_id,
                dispatch_id, queue_id_handle, correlation_id_internal,
                correlation_id_ancestor, private_segment_size, group_segment_size,
                workgroup_size_x, workgroup_size_y, workgroup_size_z, grid_size_x,
                grid_size_y, grid_size_z, stream_handle);
    }
};

struct scratch_memory_sample
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::scratch_memory;

    scratch_memory_sample() = default;
    scratch_memory_sample(std::uint64_t _start_timestamp, std::uint64_t _end_timestamp,
                          std::uint64_t _thread_id, std::uint64_t _agent_id_handle,
                          std::uint64_t _queue_id_handle, std::string_view _name,
                          std::int32_t _operation, std::int32_t _flags,
                          std::uint64_t _allocation_size,
                          std::uint64_t _correlation_id_internal,
                          std::uint64_t _correlation_id_ancestor, size_t _stream_handle)
    : start_timestamp(_start_timestamp)
    , end_timestamp(_end_timestamp)
    , thread_id(_thread_id)
    , agent_id_handle(_agent_id_handle)
    , queue_id_handle(_queue_id_handle)
    , name(_name)
    , operation(_operation)
    , flags(_flags)
    , allocation_size(_allocation_size)
    , correlation_id_internal(_correlation_id_internal)
    , correlation_id_ancestor(_correlation_id_ancestor)
    , stream_handle(_stream_handle)
    {}

    std::uint64_t    start_timestamp;
    std::uint64_t    end_timestamp;
    std::uint64_t    thread_id;
    std::uint64_t    agent_id_handle;
    std::uint64_t    queue_id_handle;
    std::string_view name;
    std::int32_t     operation;
    std::int32_t     flags;
    std::uint64_t    allocation_size;
    std::uint64_t    correlation_id_internal;
    std::uint64_t    correlation_id_ancestor;
    size_t           stream_handle;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(start_timestamp, end_timestamp, thread_id, agent_id_handle,
                queue_id_handle, name, operation, flags, allocation_size,
                correlation_id_internal, correlation_id_ancestor, stream_handle);
    }
};

struct memory_copy_sample
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::memory_copy;

    memory_copy_sample() = default;
    memory_copy_sample(std::uint64_t _start_timestamp, std::uint64_t _end_timestamp,
                       std::uint64_t _thread_id, std::uint64_t _dst_agent_id_handle,
                       std::uint64_t _src_agent_id_handle, std::string_view _name,
                       std::uint64_t _bytes, std::uint64_t _correlation_id_internal,
                       std::uint64_t _correlation_id_ancestor,
                       std::uint64_t _dst_address_value, std::uint64_t _src_address_value,
                       size_t _stream_handle)
    : start_timestamp(_start_timestamp)
    , end_timestamp(_end_timestamp)
    , thread_id(_thread_id)
    , dst_agent_id_handle(_dst_agent_id_handle)
    , src_agent_id_handle(_src_agent_id_handle)
    , name(_name)
    , bytes(_bytes)
    , correlation_id_internal(_correlation_id_internal)
    , correlation_id_ancestor(_correlation_id_ancestor)
    , dst_address_value(_dst_address_value)
    , src_address_value(_src_address_value)
    , stream_handle(_stream_handle)
    {}

    std::uint64_t    start_timestamp;
    std::uint64_t    end_timestamp;
    std::uint64_t    thread_id;
    std::uint64_t    dst_agent_id_handle;
    std::uint64_t    src_agent_id_handle;
    std::string_view name;
    std::uint64_t    bytes;
    std::uint64_t    correlation_id_internal;
    std::uint64_t    correlation_id_ancestor;
    std::uint64_t    dst_address_value;
    std::uint64_t    src_address_value;
    size_t           stream_handle;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(start_timestamp, end_timestamp, thread_id, dst_agent_id_handle,
                src_agent_id_handle, name, bytes, correlation_id_internal,
                correlation_id_ancestor, dst_address_value, src_address_value,
                stream_handle);
    }
};

struct memory_allocate_sample
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::memory_alloc;

    memory_allocate_sample() = default;
    memory_allocate_sample(std::uint64_t _start_timestamp, std::uint64_t _end_timestamp,
                           std::uint64_t _thread_id, std::uint64_t _agent_id_handle,
                           std::string_view _name, std::int32_t _operation,
                           std::uint64_t _allocation_size,
                           std::uint64_t _correlation_id_internal,
                           std::uint64_t _correlation_id_ancestor,
                           std::uint64_t _address_value, size_t _stream_handle)
    : start_timestamp(_start_timestamp)
    , end_timestamp(_end_timestamp)
    , thread_id(_thread_id)
    , agent_id_handle(_agent_id_handle)
    , name(_name)
    , operation(_operation)
    , allocation_size(_allocation_size)
    , correlation_id_internal(_correlation_id_internal)
    , correlation_id_ancestor(_correlation_id_ancestor)
    , address_value(_address_value)
    , stream_handle(_stream_handle)
    {}

    std::uint64_t    start_timestamp;
    std::uint64_t    end_timestamp;
    std::uint64_t    thread_id;
    std::uint64_t    agent_id_handle;
    std::string_view name;
    std::int32_t     operation;
    std::uint64_t    allocation_size;
    std::uint64_t    correlation_id_internal;
    std::uint64_t    correlation_id_ancestor;
    std::uint64_t    address_value;
    size_t           stream_handle;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(start_timestamp, end_timestamp, thread_id, agent_id_handle, name,
                operation, allocation_size, correlation_id_internal,
                correlation_id_ancestor, address_value, stream_handle);
    }
};

struct region_sample
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::region;

    region_sample() = default;
    region_sample(  // NOLINT(readability-function-size)
        std::uint64_t thread_id_in, std::string_view name_in,
        std::uint64_t correlation_id_internal_in,
        std::uint64_t correlation_id_ancestor_in, std::uint64_t start_timestamp_in,
        std::uint64_t end_timestamp_in, std::string_view call_stack_in,
        std::string_view args_str_in, std::string_view category_in)
    : thread_id(thread_id_in)
    , name(name_in)
    , correlation_id_internal(correlation_id_internal_in)
    , correlation_id_ancestor(correlation_id_ancestor_in)
    , start_timestamp(start_timestamp_in)
    , end_timestamp(end_timestamp_in)
    , call_stack(call_stack_in)
    , args_str(args_str_in)
    , category(category_in)
    {}

    std::uint64_t    thread_id;
    std::string_view name;
    std::uint64_t    correlation_id_internal;
    std::uint64_t    correlation_id_ancestor;
    std::uint64_t    start_timestamp;
    std::uint64_t    end_timestamp;
    std::string_view call_stack;
    std::string_view args_str;
    std::string_view category;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(thread_id, name, correlation_id_internal, correlation_id_ancestor,
                start_timestamp, end_timestamp, call_stack, args_str, category);
    }
};

struct in_time_sample
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::in_time_sample;

    in_time_sample() = default;
    in_time_sample(  // NOLINT(readability-function-size)
        size_t category_enum_id_in, std::string_view track_name_in,
        size_t timestamp_ns_in, std::string_view event_metadata_in, size_t stack_id_in,
        size_t parent_stack_id_in, size_t correlation_id_in,
        std::string_view call_stack_in, std::string_view line_info_in)
    : category_enum_id(category_enum_id_in)
    , track_name(track_name_in)
    , timestamp_ns(timestamp_ns_in)
    , event_metadata(event_metadata_in)
    , stack_id(stack_id_in)
    , parent_stack_id(parent_stack_id_in)
    , correlation_id(correlation_id_in)
    , call_stack(call_stack_in)
    , line_info(line_info_in)
    {}

    size_t           category_enum_id;
    std::string_view track_name;
    size_t           timestamp_ns;
    std::string_view event_metadata;
    size_t           stack_id;
    size_t           parent_stack_id;
    size_t           correlation_id;
    std::string_view call_stack;
    std::string_view line_info;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(category_enum_id, track_name, timestamp_ns, event_metadata, stack_id,
                parent_stack_id, correlation_id, call_stack, line_info);
    }
};

struct pmc_event_with_sample : in_time_sample
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::pmc_event_with_sample;

    pmc_event_with_sample() = default;
    pmc_event_with_sample(  // NOLINT(readability-function-size)
        size_t category_enum_id_in, std::string_view track_name_in,
        size_t timestamp_ns_in, std::string_view event_metadata_in, size_t stack_id_in,
        size_t parent_stack_id_in, size_t correlation_id_in,
        std::string_view call_stack_in, std::string_view line_info_in,
        std::uint32_t device_id_in, std::uint8_t device_type_in,
        std::string_view pmc_info_name_in, double value_in,
        std::optional<std::int64_t> system_tid_in)
    : in_time_sample(category_enum_id_in, track_name_in, timestamp_ns_in,
                     event_metadata_in, stack_id_in, parent_stack_id_in,
                     correlation_id_in, call_stack_in, line_info_in)
    , device_id(device_id_in)
    , device_type(device_type_in)
    , pmc_info_name(pmc_info_name_in)
    , value(value_in)
    , system_tid(system_tid_in)
    {}

    std::uint32_t               device_id;
    std::uint8_t                device_type;
    std::string_view            pmc_info_name;
    double                      value;
    std::optional<std::int64_t> system_tid;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        in_time_sample::serialize(archive);
        archive(device_id, device_type, pmc_info_name, value, system_tid);
    }
};

struct backtrace_region_sample
{
    static constexpr type_identifier_t type_identifier =
        type_identifier_t::backtrace_region_sample;

    backtrace_region_sample() = default;
    backtrace_region_sample(  // NOLINT(readability-function-size)
        std::uint32_t type_in, std::uint64_t thread_id_in, std::string_view track_name_in,
        std::string_view name_in, std::uint64_t start_timestamp_in,
        std::uint64_t end_timestamp_in, std::string_view category_in,
        std::string_view call_stack_in, std::string_view line_info_in,
        std::string_view extdata_in)
    : type(type_in)
    , thread_id(thread_id_in)
    , track_name(track_name_in)
    , name(name_in)
    , start_timestamp(start_timestamp_in)
    , end_timestamp(end_timestamp_in)
    , category(category_in)
    , call_stack(call_stack_in)
    , line_info(line_info_in)
    , extdata(extdata_in)
    {}

    std::uint32_t    type;
    std::uint64_t    thread_id;
    std::string_view track_name;
    std::string_view name;
    std::uint64_t    start_timestamp;
    std::uint64_t    end_timestamp;
    std::string_view category;
    std::string_view call_stack;
    std::string_view line_info;
    std::string_view extdata;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(type, thread_id, track_name, name, start_timestamp, end_timestamp,
                category, call_stack, line_info, extdata);
    }
};

struct kfd_sample
{
    static constexpr type_identifier_t type_identifier = type_identifier_t::kfd_sample;

    kfd_sample() = default;
    kfd_sample(  // NOLINT(readability-function-size)
        std::uint64_t thread_id_in, std::string_view name_in,
        std::uint64_t start_timestamp_in, std::uint64_t end_timestamp_in,
        std::string_view args_str_in, std::string_view category_in,
        std::string_view track_name_in, std::string_view event_metadata_in,
        std::uint32_t device_id_in, std::uint8_t device_type_in,
        std::string_view pmc_info_name_in, double value_in,
        std::optional<std::int64_t> system_tid_in)
    : thread_id(thread_id_in)
    , name(name_in)
    , start_timestamp(start_timestamp_in)
    , end_timestamp(end_timestamp_in)
    , args_str(args_str_in)
    , category(category_in)
    , track_name(track_name_in)
    , event_metadata(event_metadata_in)
    , device_id(device_id_in)
    , device_type(device_type_in)
    , pmc_info_name(pmc_info_name_in)
    , value(value_in)
    , system_tid(system_tid_in)
    {}

    std::uint64_t               thread_id;
    std::string_view            name;
    std::uint64_t               start_timestamp;
    std::uint64_t               end_timestamp;
    std::string_view            args_str;
    std::string_view            category;
    std::string_view            track_name;
    std::string_view            event_metadata;
    std::uint32_t               device_id;
    std::uint8_t                device_type;
    std::string_view            pmc_info_name;
    double                      value;
    std::optional<std::int64_t> system_tid;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(thread_id, name, start_timestamp, end_timestamp, args_str, category,
                track_name, event_metadata, device_id, device_type, pmc_info_name, value,
                system_tid);
    }
};

}  // namespace rocprofsys::trace_cache
