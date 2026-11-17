// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// rocpd_hip_event does not exist until schema 3.0.4.

#include "writers/hip_event_writer.hpp"
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
class hip_event_writer<data_storage::schema_v3_tag>
: public hip_event_writer_interface<hip_event_writer<data_storage::schema_v3_tag>>
{
public:
    using stmts_t =
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>;
    using common_ops_t = common_insert_operations<data_storage::schema_v3_tag>;

    explicit hip_event_writer(std::shared_ptr<writer_context> ctx,
                              std::shared_ptr<stmts_t>,
                              std::shared_ptr<common_ops_t>)
    : m_ctx(std::move(ctx))
    {}

    void insert_impl(const writer_types::hip_event_data_t&,
                     const writer_types::trace_environment_t&)
    {
        throw std::runtime_error("hip_event is not stored by schema " +
                                 m_ctx->schema_version.to_string() +
                                 ". It is stored by 3.0.4.");
    }

private:
    std::shared_ptr<writer_context> m_ctx;
};

}  // namespace profiler_hub
