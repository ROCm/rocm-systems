// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Schema 3.0.0 memory copy, also used by 3.0.1.

#include "writers/common/memory_copy_writer.hpp"
#include "writers/memory_copy_writer.hpp"

#include "data_storage/schema_version.hpp"

#include <stdexcept>
#include <string>

namespace profiler_hub
{

template <>
class memory_copy_writer<data_storage::schema_v3_tag>
: public memory_copy_common<memory_copy_writer<data_storage::schema_v3_tag>>
{
public:
    using memory_copy_common<
        memory_copy_writer<data_storage::schema_v3_tag>>::memory_copy_common;

    void reject_unsupported_fields(const writer_types::memory_copy_data_t& data) const
    {
        const auto version = m_ctx->schema_version.to_string();
        if(data.graph_exec_id.has_value())
        {
            throw std::runtime_error(
                "memory_copy.graph_exec_id is not stored by schema " + version +
                ". It is stored by 3.0.2.");
        }
        if(data.graph_node_id.has_value())
        {
            throw std::runtime_error(
                "memory_copy.graph_node_id is not stored by schema " + version +
                ". It is stored by 3.0.2.");
        }
    }

    void write_memory_copy(const writer_types::memory_copy_data_t&  data,
                           const writer_types::trace_environment_t& trace_env,
                           const memory_copy_insert&                row)
    {
        m_stmts->memory_copy_statement()(row.pk,
                                         trace_env.node_id.value(),
                                         row.process_pk,
                                         row.thread_pk,
                                         data.start_timestamp,
                                         data.end_timestamp,
                                         row.name_pk,
                                         row.dst_agent_pk,
                                         data.dst_address,
                                         row.src_agent_pk,
                                         data.src_address,
                                         data.size,
                                         row.queue_pk,
                                         row.stream_pk,
                                         row.region_name_pk,
                                         row.event_pk,
                                         data.extdata);
    }
};

}  // namespace profiler_hub
