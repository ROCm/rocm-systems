// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/output/agent_info.hpp"
#include "lib/output/node_info.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <fmt/format.h>
#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>

#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace rocpd
{
namespace types
{
namespace tool = ::rocprofiler::tool;

template <typename BaseT>
struct base_class : public BaseT
{
    using base_type = BaseT;

    auto&       base() { return static_cast<base_type&>(*this); }
    const auto& base() const { return static_cast<const base_type&>(*this); }
};

using guid_t = std::string;

// struct blob : private std::array<uint8_t, 16>
// {
//     using base_type = std::array<uint8_t, 16>;

//     auto&       base() { return static_cast<base_type&>(*this); }
//     const auto& base() const { return static_cast<const base_type&>(*this); }

//     using base_type::at;
//     using base_type::operator[];
//     using base_type::data;

//     friend bool operator==(const blob& lhs, const blob& rhs)
//     {
//         for(size_t i = 0; i < lhs.size(); ++i)
//             if(lhs.at(i) != rhs.at(i)) return false;
//         return true;
//     }

//     friend bool operator!=(const blob& lhs, const blob& rhs)
//     {
//         for(size_t i = 0; i < lhs.size(); ++i)
//             if(lhs.at(i) != rhs.at(i)) return true;
//         return false;
//     }

//     // decltype(auto) at(size_t idx) { return base_type::at(idx); }
//     // decltype(auto) at(size_t idx) const { return base_type::at(idx); }
//     // decltype(auto) operator[](size_t idx) { return
//     base_type::operator[](idx); }
//     // decltype(auto) operator[](size_t idx) const { return
//     base_type::operator[](idx); }

//     // decltype(auto) data() { return base_type::data(); }
//     // decltype(auto) data() const { return base_type::data(); }

//     std::string hexdigest() const;
//     std::string hexliteral() const;
// };

struct node : public base_class<tool::node_info>
{
    guid_t guid = {};
};

// common base class for node info
struct common_node_info
{
    guid_t      guid           = {};
    uint64_t    nid            = 0;
    std::string machine_id     = {};
    std::string hostname       = {};
    std::string system_name    = {};
    std::string system_release = {};
    std::string system_version = {};
};

struct process : public base_class<common_node_info>
{
    pid_t       ppid    = 0;
    pid_t       pid     = 0;
    uint64_t    init    = 0;
    uint64_t    start   = 0;
    uint64_t    end     = 0;
    uint64_t    fini    = 0;
    std::string command = {};
};

struct thread : public base_class<common_node_info>
{
    pid_t       ppid  = 0;
    pid_t       pid   = 0;
    pid_t       tid   = 0;
    uint64_t    start = 0;
    uint64_t    end   = 0;
    std::string name  = {};

    bool is_main_thread() const { return (tid == pid); }
};

struct agent : public base_class<tool::agent_info>
{
    guid_t      guid           = {};
    uint64_t    nid            = 0;
    uint64_t    absolute_index = 0;
    std::string type           = {};
    std::string user_name      = {};
    std::string extdata        = {};

    bool has_extdata() const { return (extdata.length() > 2); }
    void load_extdata();
};

struct code_object
{
    uint64_t    id               = 0;
    guid_t      guid             = {};
    uint64_t    nid              = 0;
    uint64_t    pid              = 0;
    uint64_t    agent_abs_index  = 0;
    std::string uri              = {};
    uint64_t    load_base        = 0;
    uint64_t    load_size        = 0;
    uint64_t    load_delta       = 0;
    std::string storage_type_str = {};
    uint64_t    storage_type     = 0;
    uint64_t    memory_base      = 0;
    uint64_t    memory_size      = 0;
    uint16_t    code_object_size = 0;
};

struct kernel_symbol
{
    uint64_t    id                            = 0;
    guid_t      guid                          = {};
    uint64_t    nid                           = 0;
    pid_t       pid                           = 0;
    uint64_t    code_object_id                = 0;
    std::string kernel_name                   = {};
    std::string display_name                  = {};
    uint64_t    kernel_id                     = 0;
    uint64_t    kernel_object                 = 0;
    uint64_t    kernarg_segment_size          = 0;
    uint64_t    kernarg_segment_alignment     = 0;
    uint64_t    group_segment_size            = 0;
    uint64_t    private_segment_size          = 0;
    uint32_t    sgpr_count                    = 0;
    uint32_t    arch_vgpr_count               = 0;
    uint32_t    accum_vgpr_count              = 0;
    uint64_t    kernel_symbol_size            = 0;
    uint64_t    kernel_code_entry_byte_offset = 0;
    std::string formatted_kernel_name         = {};
    std::string demangled_kernel_name         = {};
    std::string truncated_kernel_name         = {};
    uint64_t    kernel_address                = 0;
};

struct region
{
    struct decoded_extdata
    {
        std::string message = {};
    };

    uint64_t                id              = 0;
    guid_t                  guid            = {};
    std::string             category        = {};
    std::string             name            = {};
    pid_t                   nid             = 0;
    pid_t                   pid             = 0;
    pid_t                   tid             = 0;
    rocprofiler_timestamp_t start           = 0;
    rocprofiler_timestamp_t end             = 0;
    uint64_t                event_id        = 0;
    uint64_t                stack_id        = 0;
    uint64_t                parent_stack_id = 0;
    uint64_t                corr_id         = 0;
    std::string             extdata         = {};

    bool            has_extdata() const { return (extdata.length() > 2); }
    decoded_extdata get_extdata() const;
};

struct sample
{
    struct decoded_extdata
    {
        std::string message = {};
    };

    uint64_t                id              = 0;
    guid_t                  guid            = {};
    std::string             category        = {};
    std::string             name            = {};
    pid_t                   nid             = 0;
    pid_t                   pid             = 0;
    pid_t                   tid             = 0;
    rocprofiler_timestamp_t timestamp       = 0;
    uint64_t                event_id        = 0;
    uint64_t                stack_id        = 0;
    uint64_t                parent_stack_id = 0;
    uint64_t                corr_id         = 0;
    std::string             extdata         = {};

    bool            has_extdata() const { return (extdata.length() > 2); }
    decoded_extdata get_extdata() const;
};

struct region_arg
{
    uint64_t    id    = 0;
    guid_t      guid  = {};
    pid_t       nid   = 0;
    pid_t       pid   = 0;
    std::string type  = {};
    std::string name  = {};
    std::string value = {};
};

struct kernel_dispatch
{
    uint64_t                id                  = 0;
    guid_t                  guid                = {};
    std::string             category            = {};
    std::string             region              = {};
    std::string             name                = {};
    pid_t                   nid                 = 0;
    pid_t                   pid                 = 0;
    pid_t                   tid                 = 0;
    uint64_t                agent_abs_index     = 0;
    uint64_t                agent_log_index     = 0;
    uint64_t                agent_type_index    = 0;
    std::string             agent_type          = {};
    uint64_t                code_object_id      = 0;
    uint64_t                kernel_id           = 0;
    uint64_t                dispatch_id         = 0;
    uint64_t                stream_id           = 0;
    uint64_t                queue_id            = 0;
    std::string             queue               = {};
    std::string             stream              = {};
    rocprofiler_timestamp_t start               = 0;
    rocprofiler_timestamp_t end                 = 0;
    rocprofiler_dim3_t      grid_size           = {};
    rocprofiler_dim3_t      workgroup_size      = {};
    uint64_t                lds_size            = 0;
    uint64_t                scratch_size        = 0;
    uint64_t                static_lds_size     = 0;
    uint64_t                static_scratch_size = 0;
    uint64_t                stack_id            = 0;
    uint64_t                parent_stack_id     = 0;
    uint64_t                corr_id             = 0;
    uint64_t                vgpr_count          = 0;
    uint64_t                accum_vgpr_count    = 0;
    uint64_t                sgpr_count          = 0;
};

struct memory_allocation
{
    uint64_t                id               = 0;
    guid_t                  guid             = {};
    pid_t                   pid              = 0;
    pid_t                   tid              = 0;
    rocprofiler_timestamp_t start            = 0;
    rocprofiler_timestamp_t end              = 0;
    std::string             type             = {};
    std::string             level            = {};
    std::string             agent_name       = {};
    std::string             category         = {};
    uint64_t                agent_abs_index  = 0;
    uint64_t                agent_log_index  = 0;
    uint64_t                agent_type_index = 0;
    std::string             agent_type       = {};
    uint64_t                address          = 0;
    uint64_t                size             = 0;
    uint64_t                queue_id         = 0;
    std::string             queue_name       = {};
    uint64_t                stream_id        = 0;
    std::string             stream_name      = {};
    uint64_t                stack_id         = 0;
    uint64_t                parent_stack_id  = 0;
    uint64_t                corr_id          = 0;
};

struct memory_copies
{
    uint64_t                id                   = 0;
    guid_t                  guid                 = {};
    pid_t                   pid                  = 0;
    pid_t                   tid                  = 0;
    rocprofiler_timestamp_t start                = 0;
    rocprofiler_timestamp_t end                  = 0;
    std::string             name                 = {};
    std::string             region_name          = {};
    std::string             category             = {};
    uint64_t                stream_id            = 0;
    uint64_t                queue_id             = 0;
    std::string             stream_name          = {};
    std::string             queue_name           = {};
    uint64_t                size                 = 0;
    std::string             dst_device           = {};
    uint64_t                dst_agent_abs_index  = 0;
    uint64_t                dst_agent_log_index  = 0;
    uint64_t                dst_agent_type_index = 0;
    std::string             dst_agent_type       = {};
    uint64_t                dst_address          = 0;
    std::string             src_device           = {};
    uint64_t                src_agent_abs_index  = 0;
    uint64_t                src_agent_log_index  = 0;
    uint64_t                src_agent_type_index = 0;
    std::string             src_agent_type       = {};
    uint64_t                src_address          = 0;
    uint64_t                stack_id             = 0;
    uint64_t                parent_stack_id      = 0;
    uint64_t                corr_id              = 0;
};

struct scratch_memory
{
    guid_t                  guid             = {};
    std::string             operation        = {};
    std::string             category         = {};
    uint64_t                agent_abs_index  = 0;
    uint64_t                agent_log_index  = 0;
    uint64_t                agent_type_index = 0;
    std::string             agent_type       = {};
    uint64_t                queue_id         = 0;
    pid_t                   pid              = 0;
    pid_t                   tid              = 0;
    std::string             alloc_flags      = {};
    rocprofiler_timestamp_t start            = 0;
    rocprofiler_timestamp_t end              = 0;
    uint64_t                size             = 0;
    uint64_t                stack_id         = 0;
    uint64_t                parent_stack_id  = 0;
    uint64_t                corr_id          = 0;
};

struct stats
{
    std::string name           = {};
    uint64_t    calls          = 0;
    uint64_t    total_duration = 0;
    double      sqr            = 0.0;
    double      average        = 0.0;
    double      percentage     = 0.0;
    uint64_t    min_ns         = 0;
    uint64_t    max_ns         = 0;
    double      variance       = 0.0;
    double      std_dev        = 0.0;
};

struct stats_node
{
    guid_t guid = {};
    pid_t  pid  = 0;
    // uint64_t    nid            = 0; // nid is not used in stats_node
    std::string name           = {};
    uint64_t    calls          = 0;
    uint64_t    total_duration = 0;
    double      sqr            = 0.0;
    double      average        = 0.0;
    double      percentage     = 0.0;
    uint64_t    min_ns         = 0;
    uint64_t    max_ns         = 0;
    double      variance       = 0.0;
    double      std_dev        = 0.0;
};

// Add this struct after the existing type definitions

struct pmc_event
{
    uint64_t id            = 0;
    guid_t   guid          = {};
    pid_t    pid           = 0;
    uint64_t event_id      = 0;
    uint64_t pmc_id        = 0;
    double   counter_value = 0;
};

struct counter
{
    uint64_t                id               = 0;
    guid_t                  guid             = {};
    uint64_t                dispatch_id      = 0;
    uint64_t                kernel_id        = 0;
    uint32_t                stack_id         = 0;
    uint64_t                correlation_id   = 0;
    uint64_t                event_id         = 0;
    pid_t                   pid              = 0;
    pid_t                   tid              = 0;
    uint32_t                agent_id         = 0;
    uint64_t                agent_abs_index  = 0;
    uint64_t                agent_log_index  = 0;
    uint64_t                agent_type_index = 0;
    std::string             agent_type       = {};
    uint64_t                queue_id         = 0;
    uint32_t                grid_size_x      = 0;
    uint32_t                grid_size_y      = 0;
    uint32_t                grid_size_z      = 0;
    uint64_t                grid_size        = 0;
    std::string             kernel_name      = {};
    std::string             kernel_region    = {};
    uint32_t                workgroup_size_x = 0;
    uint32_t                workgroup_size_y = 0;
    uint32_t                workgroup_size_z = 0;
    uint32_t                workgroup_size   = 0;
    uint32_t                lds_block_size   = 0;
    uint32_t                scratch_size     = 0;
    uint32_t                vgpr_count       = 0;
    uint32_t                accum_vgpr_count = 0;
    uint32_t                sgpr_count       = 0;
    std::string             counter_name     = {};
    std::string             counter_symbol   = {};
    std::string             component        = {};
    std::string             description      = {};
    std::string             block            = {};
    std::string             expression       = {};
    std::string             value_type       = {};
    uint32_t                counter_id       = 0;
    double                  value            = 0;
    rocprofiler_timestamp_t start            = 0;
    rocprofiler_timestamp_t end              = 0;
    bool                    is_constant      = false;
    bool                    is_derived       = false;
};

struct pmc_info
{
    uint64_t    id              = 0;
    guid_t      guid            = {};
    uint64_t    nid             = 0;
    uint64_t    agent_abs_index = 0;
    bool        is_constant     = false;
    bool        is_derived      = false;
    std::string name            = {};
    std::string description     = {};
    std::string block           = {};
    std::string expression      = {};
};

}  // namespace types
}  // namespace rocpd

namespace cereal
{
#define LOAD_DATA_FIELD(FIELD)       ar(make_nvp(#FIELD, data.FIELD))
#define LOAD_DATA_NAMED(NAME, FIELD) ar(make_nvp(NAME, data.FIELD))
#define LOAD_DATA_VALUE(NAME, ARG)   ar(make_nvp(NAME, ARG))

// template <typename ArchiveT>
// void
// load(ArchiveT& ar, rocpd::types::blob& data)
// {
//     ::cereal::load(ar, data.base());
// }

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::node& data)
{
    ::cereal::load(ar, data.base());

    LOAD_DATA_FIELD(guid);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::common_node_info& data)
{
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(nid);
    LOAD_DATA_FIELD(machine_id);
    LOAD_DATA_FIELD(hostname);
    LOAD_DATA_FIELD(system_name);
    LOAD_DATA_FIELD(system_release);
    LOAD_DATA_FIELD(system_version);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::process& data)
{
    ::cereal::load(ar, data.base());

    LOAD_DATA_FIELD(ppid);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(init);
    LOAD_DATA_FIELD(start);
    LOAD_DATA_FIELD(end);
    LOAD_DATA_FIELD(fini);
    LOAD_DATA_FIELD(command);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::thread& data)
{
    ::cereal::load(ar, data.base());

    LOAD_DATA_FIELD(ppid);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(tid);
    LOAD_DATA_FIELD(start);
    LOAD_DATA_FIELD(end);
    LOAD_DATA_FIELD(name);

    if(data.name.empty()) data.name = fmt::format("Thread {}", data.tid);
    if(data.tid == data.pid && data.name.find("[main]") == std::string::npos)
        data.name += std::string(" [main]");
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::agent& data)
{
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(nid);
    LOAD_DATA_FIELD(absolute_index);
    LOAD_DATA_FIELD(type);
    LOAD_DATA_FIELD(user_name);
    LOAD_DATA_FIELD(extdata);

    data.load_extdata();
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::code_object& data)
{
    LOAD_DATA_FIELD(id);
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(nid);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(agent_abs_index);
    LOAD_DATA_FIELD(uri);
    LOAD_DATA_FIELD(load_base);
    LOAD_DATA_FIELD(load_size);
    LOAD_DATA_FIELD(load_delta);
    LOAD_DATA_FIELD(storage_type_str);
    LOAD_DATA_FIELD(storage_type);
    LOAD_DATA_FIELD(memory_base);
    LOAD_DATA_FIELD(memory_size);
    LOAD_DATA_FIELD(code_object_size);
}

// Add after the pmc_info serialization (around line 1005)

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::kernel_symbol& data)
{
    LOAD_DATA_FIELD(id);
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(nid);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(code_object_id);
    LOAD_DATA_FIELD(kernel_name);
    LOAD_DATA_FIELD(display_name);
    LOAD_DATA_FIELD(kernel_id);
    LOAD_DATA_FIELD(kernel_object);
    LOAD_DATA_FIELD(kernarg_segment_size);
    LOAD_DATA_FIELD(kernarg_segment_alignment);
    LOAD_DATA_FIELD(group_segment_size);
    LOAD_DATA_FIELD(private_segment_size);
    LOAD_DATA_FIELD(sgpr_count);
    LOAD_DATA_FIELD(arch_vgpr_count);
    LOAD_DATA_FIELD(accum_vgpr_count);
    LOAD_DATA_FIELD(kernel_symbol_size);
    LOAD_DATA_FIELD(kernel_code_entry_byte_offset);
    LOAD_DATA_FIELD(formatted_kernel_name);
    LOAD_DATA_FIELD(demangled_kernel_name);
    LOAD_DATA_FIELD(truncated_kernel_name);
    LOAD_DATA_FIELD(kernel_address);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::region& data)
{
    LOAD_DATA_FIELD(id);
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(category);
    LOAD_DATA_FIELD(name);
    LOAD_DATA_FIELD(nid);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(tid);
    LOAD_DATA_FIELD(start);
    LOAD_DATA_FIELD(end);
    LOAD_DATA_FIELD(event_id);
    LOAD_DATA_FIELD(stack_id);
    LOAD_DATA_FIELD(parent_stack_id);
    LOAD_DATA_FIELD(corr_id);
    LOAD_DATA_FIELD(extdata);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::region::decoded_extdata& data)
{
    LOAD_DATA_FIELD(message);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::sample& data)
{
    LOAD_DATA_FIELD(id);
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(category);
    LOAD_DATA_FIELD(name);
    LOAD_DATA_FIELD(nid);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(tid);
    LOAD_DATA_FIELD(timestamp);
    LOAD_DATA_FIELD(event_id);
    LOAD_DATA_FIELD(stack_id);
    LOAD_DATA_FIELD(parent_stack_id);
    LOAD_DATA_FIELD(corr_id);
    LOAD_DATA_FIELD(extdata);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::sample::decoded_extdata& data)
{
    LOAD_DATA_FIELD(message);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::region_arg& data)
{
    LOAD_DATA_FIELD(id);
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(nid);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(type);
    LOAD_DATA_FIELD(name);
    LOAD_DATA_FIELD(value);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::kernel_dispatch& data)
{
    auto load_dim3 = [&ar](std::string_view view, auto& _v) {
        ar(make_nvp(fmt::format("{}_x", view), _v.x));
        ar(make_nvp(fmt::format("{}_y", view), _v.y));
        ar(make_nvp(fmt::format("{}_z", view), _v.z));
    };

    LOAD_DATA_FIELD(id);
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(category);
    LOAD_DATA_FIELD(region);
    LOAD_DATA_FIELD(name);
    LOAD_DATA_FIELD(nid);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(tid);
    LOAD_DATA_FIELD(agent_abs_index);
    LOAD_DATA_FIELD(agent_log_index);
    LOAD_DATA_FIELD(agent_type_index);
    LOAD_DATA_FIELD(agent_type);
    LOAD_DATA_FIELD(code_object_id);
    LOAD_DATA_FIELD(kernel_id);
    LOAD_DATA_FIELD(dispatch_id);
    LOAD_DATA_FIELD(queue_id);
    LOAD_DATA_FIELD(stream_id);
    LOAD_DATA_FIELD(queue);
    LOAD_DATA_FIELD(stream);
    LOAD_DATA_FIELD(start);
    LOAD_DATA_FIELD(end);
    load_dim3("workgroup", data.workgroup_size);
    load_dim3("grid", data.grid_size);
    LOAD_DATA_FIELD(lds_size);
    LOAD_DATA_FIELD(scratch_size);
    LOAD_DATA_FIELD(vgpr_count);
    LOAD_DATA_FIELD(accum_vgpr_count);
    LOAD_DATA_FIELD(sgpr_count);
    LOAD_DATA_FIELD(static_lds_size);
    LOAD_DATA_FIELD(static_scratch_size);
    LOAD_DATA_FIELD(stack_id);
    LOAD_DATA_FIELD(parent_stack_id);
    LOAD_DATA_FIELD(corr_id);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::memory_allocation& data)
{
    LOAD_DATA_FIELD(id);
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(tid);
    LOAD_DATA_FIELD(start);
    LOAD_DATA_FIELD(end);
    LOAD_DATA_FIELD(type);
    LOAD_DATA_FIELD(level);
    LOAD_DATA_FIELD(agent_name);
    LOAD_DATA_FIELD(agent_abs_index);
    LOAD_DATA_FIELD(agent_log_index);
    LOAD_DATA_FIELD(agent_type_index);
    LOAD_DATA_FIELD(agent_type);
    LOAD_DATA_FIELD(address);
    LOAD_DATA_FIELD(size);
    LOAD_DATA_FIELD(queue_id);
    LOAD_DATA_FIELD(queue_name);
    LOAD_DATA_FIELD(stream_id);
    LOAD_DATA_FIELD(stream_name);
    LOAD_DATA_FIELD(category);
    LOAD_DATA_FIELD(stack_id);
    LOAD_DATA_FIELD(parent_stack_id);
    LOAD_DATA_FIELD(corr_id);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::memory_copies& data)
{
    LOAD_DATA_FIELD(id);
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(tid);
    LOAD_DATA_FIELD(start);
    LOAD_DATA_FIELD(end);
    LOAD_DATA_FIELD(name);
    LOAD_DATA_FIELD(region_name);
    LOAD_DATA_FIELD(stream_id);
    LOAD_DATA_FIELD(queue_id);
    LOAD_DATA_FIELD(stream_name);
    LOAD_DATA_FIELD(queue_name);
    LOAD_DATA_FIELD(size);
    LOAD_DATA_FIELD(dst_device);
    LOAD_DATA_FIELD(dst_agent_abs_index);
    LOAD_DATA_FIELD(dst_agent_log_index);
    LOAD_DATA_FIELD(dst_agent_type_index);
    LOAD_DATA_FIELD(dst_agent_type);
    LOAD_DATA_FIELD(dst_address);
    LOAD_DATA_FIELD(src_device);
    LOAD_DATA_FIELD(src_agent_abs_index);
    LOAD_DATA_FIELD(src_agent_log_index);
    LOAD_DATA_FIELD(src_agent_type_index);
    LOAD_DATA_FIELD(src_agent_type);
    LOAD_DATA_FIELD(src_address);
    LOAD_DATA_FIELD(category);
    LOAD_DATA_FIELD(stack_id);
    LOAD_DATA_FIELD(parent_stack_id);
    LOAD_DATA_FIELD(corr_id);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::scratch_memory& data)
{
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(operation);
    LOAD_DATA_FIELD(agent_abs_index);
    LOAD_DATA_FIELD(agent_log_index);
    LOAD_DATA_FIELD(agent_type_index);
    LOAD_DATA_FIELD(agent_type);
    LOAD_DATA_FIELD(queue_id);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(tid);
    LOAD_DATA_FIELD(alloc_flags);
    LOAD_DATA_FIELD(start);
    LOAD_DATA_FIELD(end);
    LOAD_DATA_FIELD(size);
    LOAD_DATA_FIELD(category);
    LOAD_DATA_FIELD(stack_id);
    LOAD_DATA_FIELD(parent_stack_id);
    LOAD_DATA_FIELD(corr_id);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::stats& data)
{
    LOAD_DATA_FIELD(name);
    LOAD_DATA_FIELD(calls);
    LOAD_DATA_NAMED("DURATION (nsec)", total_duration);
    LOAD_DATA_NAMED("SQR (nsec)", sqr);
    LOAD_DATA_NAMED("AVERAGE (nsec)", average);
    LOAD_DATA_NAMED("PERCENT (INC)", percentage);
    LOAD_DATA_NAMED("MIN (nsec)", min_ns);
    LOAD_DATA_NAMED("MAX (nsec)", max_ns);
    LOAD_DATA_NAMED("VARIANCE", variance);
    LOAD_DATA_NAMED("STD_DEV", std_dev);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::stats_node& data)
{
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(pid);
    // LOAD_DATA_FIELD(nid); // nid is not used in stats_node
    LOAD_DATA_FIELD(name);
    LOAD_DATA_FIELD(calls);
    LOAD_DATA_NAMED("DURATION (nsec)", total_duration);
    LOAD_DATA_NAMED("SQR (nsec)", sqr);
    LOAD_DATA_NAMED("AVERAGE (nsec)", average);
    LOAD_DATA_NAMED("PERCENT (INC)", percentage);
    LOAD_DATA_NAMED("MIN (nsec)", min_ns);
    LOAD_DATA_NAMED("MAX (nsec)", max_ns);
    LOAD_DATA_NAMED("VARIANCE", variance);
    LOAD_DATA_NAMED("STD_DEV", std_dev);
}

// Add this inside the cereal namespace, after the existing load functions

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::pmc_event& data)
{
    LOAD_DATA_FIELD(id);
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(event_id);
    LOAD_DATA_FIELD(pmc_id);
    // LOAD_DATA_FIELD(counter_value);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::counter& data)
{
    LOAD_DATA_FIELD(id);
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(dispatch_id);
    LOAD_DATA_FIELD(kernel_id);
    LOAD_DATA_FIELD(stack_id);
    LOAD_DATA_FIELD(correlation_id);
    LOAD_DATA_FIELD(event_id);
    LOAD_DATA_FIELD(pid);
    LOAD_DATA_FIELD(tid);
    LOAD_DATA_FIELD(agent_id);
    LOAD_DATA_FIELD(agent_abs_index);
    LOAD_DATA_FIELD(agent_log_index);
    LOAD_DATA_FIELD(agent_type_index);
    LOAD_DATA_FIELD(agent_type);
    LOAD_DATA_FIELD(queue_id);
    LOAD_DATA_FIELD(grid_size_x);
    LOAD_DATA_FIELD(grid_size_y);
    LOAD_DATA_FIELD(grid_size_z);
    LOAD_DATA_FIELD(grid_size);
    LOAD_DATA_FIELD(kernel_name);
    LOAD_DATA_FIELD(kernel_region);
    LOAD_DATA_FIELD(workgroup_size_x);
    LOAD_DATA_FIELD(workgroup_size_y);
    LOAD_DATA_FIELD(workgroup_size_z);
    LOAD_DATA_FIELD(workgroup_size);
    LOAD_DATA_FIELD(lds_block_size);
    LOAD_DATA_FIELD(scratch_size);
    LOAD_DATA_FIELD(vgpr_count);
    LOAD_DATA_FIELD(accum_vgpr_count);
    LOAD_DATA_FIELD(sgpr_count);
    LOAD_DATA_FIELD(counter_name);
    LOAD_DATA_FIELD(counter_symbol);
    LOAD_DATA_FIELD(component);
    LOAD_DATA_FIELD(description);
    LOAD_DATA_FIELD(block);
    LOAD_DATA_FIELD(expression);
    LOAD_DATA_FIELD(value_type);
    LOAD_DATA_FIELD(counter_id);
    LOAD_DATA_FIELD(value);
    LOAD_DATA_FIELD(start);
    LOAD_DATA_FIELD(end);
    LOAD_DATA_FIELD(is_constant);
    LOAD_DATA_FIELD(is_derived);
}
template <typename ArchiveT>
void
load(ArchiveT& ar, rocpd::types::pmc_info& data)
{
    LOAD_DATA_FIELD(id);
    LOAD_DATA_FIELD(guid);
    LOAD_DATA_FIELD(nid);
    LOAD_DATA_FIELD(agent_abs_index);
    LOAD_DATA_FIELD(is_constant);
    LOAD_DATA_FIELD(is_derived);
    LOAD_DATA_FIELD(name);
    LOAD_DATA_FIELD(description);
    LOAD_DATA_FIELD(block);
    LOAD_DATA_FIELD(expression);
}

}  // namespace cereal

#undef LOAD_DATA_FIELD
#undef LOAD_DATA_NAMED
#undef LOAD_DATA_VALUE

// namespace fmt
// {
// template <>
// struct formatter<rocpd::types::blob>
// {
//     template <typename ParseContext>
//     constexpr auto parse(ParseContext& ctx)
//     {
//         return ctx.begin();
//     }

//     template <typename Ctx>
//     auto format(const rocpd::types::blob& val, Ctx& ctx) const
//     {
//         return fmt::format_to(ctx.out(), "{}", val.hexliteral());
//     }
// };
// }  // namespace fmt
