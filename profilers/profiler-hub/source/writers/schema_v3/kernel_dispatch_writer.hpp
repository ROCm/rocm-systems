// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Schema 3.0.0 kernel dispatch, also used by 3.0.1. graph_exec_id and
// graph_node_id are not columns until 3.0.2.

#include "writers/common/kernel_dispatch_writer.hpp"
#include "writers/kernel_dispatch_writer.hpp"

#include "data_storage/schema_version.hpp"

#include <stdexcept>
#include <string>

namespace profiler_hub
{

template <>
class kernel_dispatch_writer<data_storage::schema_v3_tag>
: public kernel_dispatch_common<kernel_dispatch_writer<data_storage::schema_v3_tag>>
{
public:
    using kernel_dispatch_common<
        kernel_dispatch_writer<data_storage::schema_v3_tag>>::kernel_dispatch_common;

    void reject_unsupported_fields(const writer_types::kernel_dispatch_data_t& data) const
    {
        const auto version = m_ctx->schema_version.to_string();
        if(data.graph_exec_id.has_value())
        {
            throw std::runtime_error(
                "kernel_dispatch.graph_exec_id is not stored by schema " + version +
                ". It is stored by 3.0.2.");
        }
        if(data.graph_node_id.has_value())
        {
            throw std::runtime_error(
                "kernel_dispatch.graph_node_id is not stored by schema " + version +
                ". It is stored by 3.0.2.");
        }
    }

    void write_kernel_dispatch(const writer_types::kernel_dispatch_data_t& data,
                               const writer_types::trace_environment_t&    trace_env,
                               const kernel_dispatch_insert&               row)
    {
        m_stmts->kernel_dispatch_statement()(row.pk,
                                             trace_env.node_id.value(),
                                             row.process_pk,
                                             row.thread_pk,
                                             row.agent_pk,
                                             data.kernel_symbol_id,
                                             data.dispatch_id,
                                             row.queue_pk,
                                             row.stream_pk,
                                             data.start_timestamp,
                                             data.end_timestamp,
                                             data.private_segment_size,
                                             data.group_segment_size,
                                             data.workgroup_size_x,
                                             data.workgroup_size_y,
                                             data.workgroup_size_z,
                                             data.grid_size_x,
                                             data.grid_size_y,
                                             data.grid_size_z,
                                             row.name_pk,
                                             row.event_pk,
                                             data.extdata);
    }
};

}  // namespace profiler_hub
