// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// rocpd_graph_launch does not exist until schema 3.0.2.

#include "writers/graph_launch_writer.hpp"
#include "writers/schema_v3/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v3/insert_statements.hpp"
#include "data_storage/schema_version.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace profiler_hub
{

template <>
class graph_launch_writer<data_storage::schema_v3_tag>
: public graph_launch_writer_interface<graph_launch_writer<data_storage::schema_v3_tag>>
{
public:
    using stmts_t =
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>;
    using common_ops_t = common_insert_operations<data_storage::schema_v3_tag>;

    explicit graph_launch_writer(std::shared_ptr<writer_context> ctx,
                                 std::shared_ptr<stmts_t>,
                                 std::shared_ptr<common_ops_t>)
    : m_ctx(std::move(ctx))
    {}

    void insert_impl(const writer_types::graph_launch_data_t&,
                     const writer_types::trace_environment_t&)
    {
        throw std::runtime_error("graph_launch is not stored by schema " +
                                 m_ctx->schema_version.to_string() +
                                 ". It is stored by 3.0.2.");
    }

private:
    std::shared_ptr<writer_context> m_ctx;
};

}  // namespace profiler_hub
