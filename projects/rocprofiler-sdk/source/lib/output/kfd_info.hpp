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

#include "metadata.hpp"

#include <map>

namespace rocprofiler
{
namespace tool
{
static auto
agent_node_id(const metadata& tool_metadata, const rocprofiler_agent_id_t& agent)
{
    return agent.handle == 0 ? -1 : int32_t(tool_metadata.get_agent(agent)->node_id);
}

// rocpd_kfd_event_data is for encoding arbitrary kfd event data that does not
// map well into structured rocpd tables. The primary use of this type is to
// serialize into rocpd_event's extdata column.
//
// The structure is simply a map of string:string, which makes parsing on the
// rocpd side simpler by limiting how much of the kfd implementation internals
// need to make it into the rocpd::types structures

using rocpd_kfd_event_data = std::map<std::string, std::string>;

// The types defined below are simple wrappers of the
// rocprofiler_buffer_tracing_kfd_... types, and are defined so that cereal's
// save() routines invoked on them will lead to proper serialization through the
// string-based rocpd_kfd_event_data type

struct rocpd_kfd_event_page_migrate_record_t
: rocprofiler_buffer_tracing_kfd_event_page_migrate_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_event_page_migrate_record_t;

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

    uint64_t start_address;
    uint64_t end_address;
    int64_t  src_agent_id;
    int64_t  dst_agent_id;
    int64_t  prefetch_agent_id;
    int64_t  preferred_agent_id;
};

struct rocpd_kfd_event_page_fault_record_t
: rocprofiler_buffer_tracing_kfd_event_page_fault_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_event_page_fault_record_t;

    rocpd_kfd_event_page_fault_record_t(const base_type& _base, const metadata& _metadata)
    : base_type(_base)
    {
        agent_id = agent_node_id(_metadata, _base.agent_id);
        address  = _base.address.value;
    }

    int64_t  agent_id;
    uint64_t address;
};

struct rocpd_kfd_event_queue_record_t : rocprofiler_buffer_tracing_kfd_event_queue_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_event_queue_record_t;

    rocpd_kfd_event_queue_record_t(const base_type& _base, const metadata& _metadata)
    : base_type(_base)
    {
        agent_id = agent_node_id(_metadata, _base.agent_id);
    }

    int64_t agent_id;
};

struct rocpd_kfd_event_unmap_from_gpu_record_t
: rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t;

    rocpd_kfd_event_unmap_from_gpu_record_t(const base_type& _base, const metadata& _metadata)
    : base_type(_base)
    {
        agent_id      = agent_node_id(_metadata, _base.agent_id);
        start_address = _base.start_address.value;
        end_address   = _base.end_address.value;
    }

    int64_t  agent_id;
    uint64_t start_address;
    uint64_t end_address;
};

struct rocpd_kfd_event_dropped_events_record_t
: rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t;

    rocpd_kfd_event_dropped_events_record_t(const base_type& _base, const metadata& /*_metadata*/)
    : base_type(_base)
    {}
};

struct rocpd_kfd_page_migrate_record_t : rocprofiler_buffer_tracing_kfd_page_migrate_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_page_migrate_record_t;

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

    uint64_t start_address;
    uint64_t end_address;
    int64_t  src_agent_id;
    int64_t  dst_agent_id;
    int64_t  prefetch_agent_id;
    int64_t  preferred_agent_id;
};

struct rocpd_kfd_page_fault_record_t : rocprofiler_buffer_tracing_kfd_page_fault_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_page_fault_record_t;

    rocpd_kfd_page_fault_record_t(const base_type& _base, const metadata& _metadata)
    : base_type(_base)
    {
        agent_id = agent_node_id(_metadata, _base.agent_id);
        address  = _base.address.value;
    }

    int64_t  agent_id;
    uint64_t address;
};

struct rocpd_kfd_queue_record_t : rocprofiler_buffer_tracing_kfd_queue_record_t
{
    using base_type = rocprofiler_buffer_tracing_kfd_queue_record_t;

    rocpd_kfd_queue_record_t(const base_type& _base, const metadata& _metadata)
    : base_type(_base)
    {
        agent_id = agent_node_id(_metadata, _base.agent_id);
    }

    int64_t agent_id;
};

}  // namespace tool
}  // namespace rocprofiler

namespace cereal
{
#define KFD_FIELD(FIELD)                                                                           \
    {                                                                                              \
#        FIELD, std::to_string(rec.FIELD)                                                          \
    }

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_event_page_migrate_record_t& rec)
{
    ar(cereal::make_nvp("kfd",
                        ::rocprofiler::tool::rocpd_kfd_event_data({
                            KFD_FIELD(start_address),
                            KFD_FIELD(end_address),
                            KFD_FIELD(src_agent_id),
                            KFD_FIELD(dst_agent_id),
                            KFD_FIELD(prefetch_agent_id),
                            KFD_FIELD(preferred_agent_id),
                            KFD_FIELD(error_code),
                        })));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_event_page_fault_record_t& rec)
{
    ar(cereal::make_nvp("kfd",
                        ::rocprofiler::tool::rocpd_kfd_event_data({
                            KFD_FIELD(agent_id),
                            KFD_FIELD(address),
                        })));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_event_queue_record_t& rec)
{
    ar(cereal::make_nvp("kfd",
                        ::rocprofiler::tool::rocpd_kfd_event_data({
                            KFD_FIELD(agent_id),
                        })));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_event_unmap_from_gpu_record_t& rec)
{
    ar(cereal::make_nvp("kfd",
                        ::rocprofiler::tool::rocpd_kfd_event_data({
                            KFD_FIELD(agent_id),
                            KFD_FIELD(start_address),
                            KFD_FIELD(end_address),
                        })));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_event_dropped_events_record_t& rec)
{
    ar(cereal::make_nvp("kfd",
                        ::rocprofiler::tool::rocpd_kfd_event_data({
                            KFD_FIELD(count),
                        })));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_page_migrate_record_t& rec)
{
    ar(cereal::make_nvp("kfd",
                        ::rocprofiler::tool::rocpd_kfd_event_data({
                            KFD_FIELD(start_address),
                            KFD_FIELD(end_address),
                            KFD_FIELD(src_agent_id),
                            KFD_FIELD(dst_agent_id),
                            KFD_FIELD(prefetch_agent_id),
                            KFD_FIELD(preferred_agent_id),
                            KFD_FIELD(error_code),
                        })));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_page_fault_record_t& rec)
{
    ar(cereal::make_nvp("kfd",
                        ::rocprofiler::tool::rocpd_kfd_event_data({
                            KFD_FIELD(agent_id),
                            KFD_FIELD(address),
                        })));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::rocpd_kfd_queue_record_t& rec)
{
    ar(cereal::make_nvp("kfd",
                        ::rocprofiler::tool::rocpd_kfd_event_data({
                            KFD_FIELD(agent_id),
                        })));
}

#undef KFD_FIELD

}  // namespace cereal
