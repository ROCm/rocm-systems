// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "core/trace_cache/archive.hpp"
#include "core/trace_cache/cache_type_traits.hpp"
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

    // stream_handle is widened to uint64 for portability across 32 / 64-bit
    // builds; this matches the static_cast in the original wire format.
    //
    // ROCPROFSYS_INLINE expands to `[[gnu::always_inline]] inline`. Combined
    // with ROCPROFSYS_FLATTEN on the serialize_to/serialized_size entry
    // points this collapses the per-field memcpy chain into the caller's
    // frame. Without it GCC keeps this template instantiation as an
    // out-of-line call and reloads the cursor between fields.
    template <class Archive>
    ROCPROFSYS_INLINE void serialize(Archive& ar)
    {
        auto stream_handle_u64 = static_cast<std::uint64_t>(stream_handle);
        ar(start_timestamp, end_timestamp, thread_id, agent_id_handle, kernel_id,
           dispatch_id, queue_id_handle, correlation_id_internal, correlation_id_ancestor,
           private_segment_size, group_segment_size, workgroup_size_x, workgroup_size_y,
           workgroup_size_z, grid_size_x, grid_size_y, grid_size_z, stream_handle_u64);
        if constexpr(std::is_same_v<Archive, input_archive>)
        {
            stream_handle = static_cast<std::size_t>(stream_handle_u64);
        }
    }
};

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

    template <class Archive>
    ROCPROFSYS_INLINE void serialize(Archive& ar)
    {
        auto stream_handle_u64 = static_cast<std::uint64_t>(stream_handle);
        ar(start_timestamp, end_timestamp, thread_id, agent_id_handle, queue_id_handle,
           kind, operation, flags, allocation_size, correlation_id_internal,
           correlation_id_ancestor, stream_handle_u64);
        if constexpr(std::is_same_v<Archive, input_archive>)
        {
            stream_handle = static_cast<std::size_t>(stream_handle_u64);
        }
    }
};

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

    template <class Archive>
    ROCPROFSYS_INLINE void serialize(Archive& ar)
    {
        auto stream_handle_u64 = static_cast<std::uint64_t>(stream_handle);
        ar(start_timestamp, end_timestamp, thread_id, dst_agent_id_handle,
           src_agent_id_handle, kind, operation, bytes, correlation_id_internal,
           correlation_id_ancestor, dst_address_value, src_address_value,
           stream_handle_u64);
        if constexpr(std::is_same_v<Archive, input_archive>)
        {
            stream_handle = static_cast<std::size_t>(stream_handle_u64);
        }
    }
};

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

    template <class Archive>
    ROCPROFSYS_INLINE void serialize(Archive& ar)
    {
        auto stream_handle_u64 = static_cast<std::uint64_t>(stream_handle);
        ar(start_timestamp, end_timestamp, thread_id, agent_id_handle, kind, operation,
           allocation_size, correlation_id_internal, correlation_id_ancestor,
           address_value, stream_handle_u64);
        if constexpr(std::is_same_v<Archive, input_archive>)
        {
            stream_handle = static_cast<std::size_t>(stream_handle_u64);
        }
    }
};

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

    template <class Archive>
    ROCPROFSYS_INLINE void serialize(Archive& ar)
    {
        ar(thread_id, name, correlation_id_internal, correlation_id_ancestor,
           start_timestamp, end_timestamp, call_stack, args_str, category);
    }
};

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

    template <class Archive>
    ROCPROFSYS_INLINE void serialize(Archive& ar)
    {
        // category_enum_id is left as size_t to match legacy free serialize
        // (not widened); other size_t members are widened to uint64 for wire
        // portability, mirroring the static_cast in the legacy path.
        auto timestamp_ns_u64    = static_cast<std::uint64_t>(timestamp_ns);
        auto stack_id_u64        = static_cast<std::uint64_t>(stack_id);
        auto parent_stack_id_u64 = static_cast<std::uint64_t>(parent_stack_id);
        auto correlation_id_u64  = static_cast<std::uint64_t>(correlation_id);
        ar(category_enum_id, track_name, timestamp_ns_u64, event_metadata, stack_id_u64,
           parent_stack_id_u64, correlation_id_u64, call_stack, line_info);
        if constexpr(std::is_same_v<Archive, input_archive>)
        {
            timestamp_ns    = static_cast<std::size_t>(timestamp_ns_u64);
            stack_id        = static_cast<std::size_t>(stack_id_u64);
            parent_stack_id = static_cast<std::size_t>(parent_stack_id_u64);
            correlation_id  = static_cast<std::size_t>(correlation_id_u64);
        }
    }
};

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

    template <class Archive>
    ROCPROFSYS_INLINE void serialize(Archive& ar)
    {
        // Match legacy: parent fields field-by-field, then derived fields. Do
        // not delegate to the parent serialize<Archive> because that detection
        // would still match here on the derived type and would also re-emit
        // the parent fields exactly as needed - but explicit listing keeps
        // the wire layout obvious next to the legacy free function.
        auto timestamp_ns_u64    = static_cast<std::uint64_t>(timestamp_ns);
        auto stack_id_u64        = static_cast<std::uint64_t>(stack_id);
        auto parent_stack_id_u64 = static_cast<std::uint64_t>(parent_stack_id);
        auto correlation_id_u64  = static_cast<std::uint64_t>(correlation_id);
        ar(category_enum_id, track_name, timestamp_ns_u64, event_metadata, stack_id_u64,
           parent_stack_id_u64, correlation_id_u64, call_stack, line_info, device_id,
           device_type, pmc_info_name, value, system_tid);
        if constexpr(std::is_same_v<Archive, input_archive>)
        {
            timestamp_ns    = static_cast<std::size_t>(timestamp_ns_u64);
            stack_id        = static_cast<std::size_t>(stack_id_u64);
            parent_stack_id = static_cast<std::size_t>(parent_stack_id_u64);
            correlation_id  = static_cast<std::size_t>(correlation_id_u64);
        }
    }
};

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

    template <class Archive>
    ROCPROFSYS_INLINE void serialize(Archive& ar)
    {
        ar(type, thread_id, track_name, name, start_timestamp, end_timestamp, category,
           call_stack, line_info, extdata);
    }
};

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

    template <class Archive>
    ROCPROFSYS_INLINE void serialize(Archive& ar)
    {
        ar(thread_id, name, start_timestamp, end_timestamp, args_str, category,
           track_name, event_metadata, device_id, device_type, pmc_info_name, value,
           system_tid);
    }
};

}  // namespace trace_cache
}  // namespace rocprofsys
