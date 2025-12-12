// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/mpl.hpp"
#include "metadata.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <variant>

namespace rocprofiler
{
namespace tool
{
// Wrapper type used by the tool to simplify the passing around of KFD data
struct tool_buffer_tracing_kfd_record_t
{
    // use std::monostate to indicate no KFD data present
    using data_type = std::variant<std::monostate,
                                   rocprofiler_buffer_tracing_kfd_event_page_migrate_record_t,
                                   rocprofiler_buffer_tracing_kfd_event_page_fault_record_t,
                                   rocprofiler_buffer_tracing_kfd_event_queue_record_t,
                                   rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t,
                                   rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t,
                                   rocprofiler_buffer_tracing_kfd_page_migrate_record_t,
                                   rocprofiler_buffer_tracing_kfd_page_fault_record_t,
                                   rocprofiler_buffer_tracing_kfd_queue_record_t>;

    data_type record = {};  // default initializes to monostate

    bool valid() const { return !std::holds_alternative<std::monostate>(record); }
};

inline std::optional<uint32_t>
agent_node_id(const metadata& tool_metadata, rocprofiler_agent_id_t agent)
{
    const auto* _agent = tool_metadata.get_agent(agent);
    return (_agent) ? std::optional<uint32_t>{_agent->node_id} : std::nullopt;
}

// The rocpd_kfd_* types defined below are simple wrappers of the
// rocprofiler_buffer_tracing_kfd_... types, and are defined so that cereal's
// save() routines invoked on them will lead to proper serializetion of miscellaneous
// data into extdata JSON strings in rocpd tables

struct rocpd_kfd_event_page_migrate_record_t
: rocprofiler_buffer_tracing_kfd_event_page_migrate_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_event_page_migrate_record_t;

    rocpd_kfd_event_page_migrate_record_t() = default;
    rocpd_kfd_event_page_migrate_record_t(const base_type& _base, const metadata& _metadata)
    : base_type(_base)
    {
        start_address      = _base.start_address.value;
        end_address        = _base.end_address.value;
        src_agent_id       = agent_node_id(_metadata, _base.src_agent);
        dst_agent_id       = agent_node_id(_metadata, _base.dst_agent);
        prefetch_agent_id  = agent_node_id(_metadata, _base.prefetch_agent);
        preferred_agent_id = agent_node_id(_metadata, _base.preferred_agent);
    }

    uint64_t value() const { return end_address - start_address; }

    uint64_t                start_address      = 0;
    uint64_t                end_address        = 0;
    std::optional<uint32_t> src_agent_id       = std::nullopt;
    std::optional<uint32_t> dst_agent_id       = std::nullopt;
    std::optional<uint32_t> prefetch_agent_id  = std::nullopt;
    std::optional<uint32_t> preferred_agent_id = std::nullopt;
};

struct rocpd_kfd_event_page_fault_record_t
: rocprofiler_buffer_tracing_kfd_event_page_fault_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_event_page_fault_record_t;

    rocpd_kfd_event_page_fault_record_t() = default;
    rocpd_kfd_event_page_fault_record_t(const base_type& _base, const metadata& _metadata)
    : base_type(_base)
    {
        agent_id = agent_node_id(_metadata, _base.agent_id);
        address  = _base.address.value;
    }

    uint64_t value() const { return address; }

    std::optional<uint32_t> agent_id = std::nullopt;
    uint64_t                address  = 0;
};

struct rocpd_kfd_event_queue_record_t : rocprofiler_buffer_tracing_kfd_event_queue_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_event_queue_record_t;

    rocpd_kfd_event_queue_record_t() = default;
    rocpd_kfd_event_queue_record_t(const base_type& _base, const metadata& _metadata)
    : base_type(_base)
    {
        agent_id = agent_node_id(_metadata, _base.agent_id);
    }

    static uint64_t value() { return 1; }

    std::optional<uint32_t> agent_id = std::nullopt;
};

struct rocpd_kfd_event_unmap_from_gpu_record_t
: rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t;

    rocpd_kfd_event_unmap_from_gpu_record_t() = default;
    rocpd_kfd_event_unmap_from_gpu_record_t(const base_type& _base, const metadata& _metadata)
    : base_type(_base)
    {
        agent_id      = agent_node_id(_metadata, _base.agent_id);
        start_address = _base.start_address.value;
        end_address   = _base.end_address.value;
    }

    uint64_t value() const { return end_address - start_address; }

    std::optional<uint32_t> agent_id      = std::nullopt;
    uint64_t                start_address = 0;
    uint64_t                end_address   = 0;
};

struct rocpd_kfd_event_dropped_events_record_t
: rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t;

    rocpd_kfd_event_dropped_events_record_t() = default;
    rocpd_kfd_event_dropped_events_record_t(const base_type& _base, const metadata& /*_metadata*/)
    : base_type(_base)
    {}

    uint64_t value() const { return count; }
};

struct rocpd_kfd_page_migrate_record_t : rocprofiler_buffer_tracing_kfd_page_migrate_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_page_migrate_record_t;

    rocpd_kfd_page_migrate_record_t() = default;
    rocpd_kfd_page_migrate_record_t(const base_type& _base, const metadata& _metadata)
    : base_type(_base)
    {
        start_address      = _base.start_address.value;
        end_address        = _base.end_address.value;
        src_agent_id       = agent_node_id(_metadata, _base.src_agent);
        dst_agent_id       = agent_node_id(_metadata, _base.dst_agent);
        prefetch_agent_id  = agent_node_id(_metadata, _base.prefetch_agent);
        preferred_agent_id = agent_node_id(_metadata, _base.preferred_agent);
    }

    uint64_t value() const { return end_address - start_address; }

    uint64_t                start_address      = 0;
    uint64_t                end_address        = 0;
    std::optional<uint32_t> src_agent_id       = std::nullopt;
    std::optional<uint32_t> dst_agent_id       = std::nullopt;
    std::optional<uint32_t> prefetch_agent_id  = std::nullopt;
    std::optional<uint32_t> preferred_agent_id = std::nullopt;
};

struct rocpd_kfd_page_fault_record_t : rocprofiler_buffer_tracing_kfd_page_fault_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_page_fault_record_t;

    rocpd_kfd_page_fault_record_t() = default;
    rocpd_kfd_page_fault_record_t(const base_type& _base, const metadata& _metadata)
    : base_type(_base)
    {
        agent_id = agent_node_id(_metadata, _base.agent_id);
        address  = _base.address.value;
    }

    uint64_t value() const { return address; }

    std::optional<uint32_t> agent_id = std::nullopt;
    uint64_t                address  = 0;
};

struct rocpd_kfd_queue_record_t : rocprofiler_buffer_tracing_kfd_queue_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_queue_record_t;

    rocpd_kfd_queue_record_t() = default;
    rocpd_kfd_queue_record_t(const base_type& _base, const metadata& _metadata)
    : base_type(_base)
    {
        agent_id = agent_node_id(_metadata, _base.agent_id);
    }

    static uint64_t value() { return 1; }

    std::optional<uint32_t> agent_id = std::nullopt;
};

// A variant capturing all of the rocpd_kfd_* types
struct rocpd_kfd_event_data_t
{
    // use std::monostate to indicate no KFD data present
    using data_type = std::variant<std::monostate,
                                   rocpd_kfd_event_page_migrate_record_t,
                                   rocpd_kfd_event_page_fault_record_t,
                                   rocpd_kfd_event_queue_record_t,
                                   rocpd_kfd_event_unmap_from_gpu_record_t,
                                   rocpd_kfd_event_dropped_events_record_t,
                                   rocpd_kfd_page_migrate_record_t,
                                   rocpd_kfd_page_fault_record_t,
                                   rocpd_kfd_queue_record_t>;

    data_type record = {};  // default initialized to monostate
};
}  // namespace tool
}  // namespace rocprofiler

namespace cereal
{
#define SAVE_DATA_FIELD(FIELD) ar(cereal::make_nvp(#FIELD, rec.FIELD))
#define LOAD_DATA_FIELD(FIELD) ar(cereal::make_nvp(#FIELD, rec.FIELD))
#define SAVE_OPT_DATA_FIELD(FIELD)                                                                 \
    if(rec.FIELD) ar(cereal::make_nvp(#FIELD, *rec.FIELD))

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::tool_buffer_tracing_kfd_record_t& data)
{
    std::visit(
        [&ar](const auto record) {
            using record_type = ::rocprofiler::common::mpl::unqualified_type_t<decltype(record)>;

            if constexpr(!std::is_same<record_type, std::monostate>::value)
                cereal::save(ar, record);
        },
        data.record);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_event_page_migrate_record_t& rec)
{
    SAVE_DATA_FIELD(start_address);
    SAVE_DATA_FIELD(end_address);
    SAVE_OPT_DATA_FIELD(src_agent_id);
    SAVE_OPT_DATA_FIELD(dst_agent_id);
    SAVE_OPT_DATA_FIELD(prefetch_agent_id);
    SAVE_OPT_DATA_FIELD(preferred_agent_id);
    SAVE_DATA_FIELD(error_code);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_event_page_fault_record_t& rec)
{
    SAVE_OPT_DATA_FIELD(agent_id);
    SAVE_DATA_FIELD(address);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_event_queue_record_t& rec)
{
    SAVE_OPT_DATA_FIELD(agent_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_event_unmap_from_gpu_record_t& rec)
{
    SAVE_OPT_DATA_FIELD(agent_id);
    SAVE_DATA_FIELD(start_address);
    SAVE_DATA_FIELD(end_address);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_event_dropped_events_record_t& rec)
{
    SAVE_DATA_FIELD(count);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_page_migrate_record_t& rec)
{
    SAVE_DATA_FIELD(start_address);
    SAVE_DATA_FIELD(end_address);
    SAVE_OPT_DATA_FIELD(src_agent_id);
    SAVE_OPT_DATA_FIELD(dst_agent_id);
    SAVE_OPT_DATA_FIELD(prefetch_agent_id);
    SAVE_OPT_DATA_FIELD(preferred_agent_id);
    SAVE_DATA_FIELD(error_code);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_page_fault_record_t& rec)
{
    SAVE_OPT_DATA_FIELD(agent_id);
    SAVE_DATA_FIELD(address);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_queue_record_t& rec)
{
    SAVE_OPT_DATA_FIELD(agent_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_event_data_t& rec)
{
    SAVE_DATA_FIELD(record);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, ::rocprofiler::tool::rocpd_kfd_event_page_migrate_record_t& rec)
{
    LOAD_DATA_FIELD(start_address);
    LOAD_DATA_FIELD(end_address);
    LOAD_DATA_FIELD(src_agent_id);
    LOAD_DATA_FIELD(dst_agent_id);
    LOAD_DATA_FIELD(prefetch_agent_id);
    LOAD_DATA_FIELD(preferred_agent_id);
    LOAD_DATA_FIELD(error_code);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, ::rocprofiler::tool::rocpd_kfd_event_page_fault_record_t& rec)
{
    LOAD_DATA_FIELD(agent_id);
    LOAD_DATA_FIELD(address);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, ::rocprofiler::tool::rocpd_kfd_event_queue_record_t& rec)
{
    LOAD_DATA_FIELD(agent_id);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, ::rocprofiler::tool::rocpd_kfd_event_unmap_from_gpu_record_t& rec)
{
    LOAD_DATA_FIELD(agent_id);
    LOAD_DATA_FIELD(start_address);
    LOAD_DATA_FIELD(end_address);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, ::rocprofiler::tool::rocpd_kfd_event_dropped_events_record_t& rec)
{
    LOAD_DATA_FIELD(count);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, ::rocprofiler::tool::rocpd_kfd_page_migrate_record_t& rec)
{
    LOAD_DATA_FIELD(start_address);
    LOAD_DATA_FIELD(end_address);
    LOAD_DATA_FIELD(src_agent_id);
    LOAD_DATA_FIELD(dst_agent_id);
    LOAD_DATA_FIELD(prefetch_agent_id);
    LOAD_DATA_FIELD(preferred_agent_id);
    LOAD_DATA_FIELD(error_code);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, ::rocprofiler::tool::rocpd_kfd_page_fault_record_t& rec)
{
    LOAD_DATA_FIELD(agent_id);
    LOAD_DATA_FIELD(address);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, ::rocprofiler::tool::rocpd_kfd_queue_record_t& rec)
{
    LOAD_DATA_FIELD(agent_id);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, ::rocprofiler::tool::rocpd_kfd_event_data_t& rec)
{
    LOAD_DATA_FIELD(record);
}

#undef SAVE_DATA_FIELD
#undef LOAD_DATA_FIELD

}  // namespace cereal
