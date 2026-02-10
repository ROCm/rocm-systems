// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "writer_impl.hpp"
#include "storage_impl.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace rocstorage
{

writer_t::impl::impl(std::unique_ptr<rocstorage::storage_t> storage)
: m_storage(std::move(storage))
, m_ctx(std::make_shared<writer_context>())
{
    m_ctx->backend =
        m_storage->m_impl->create_database(storage_t::impl::storage_type_t::write);
    m_ctx->registry      = std::make_shared<entity_registry>();
    m_ctx->key_providers = std::make_shared<primary_key_providers>();
    m_ctx->uuid          = m_storage->m_impl->get_uuid();

    if(!m_ctx->backend)
    {
        throw std::invalid_argument("Provided pointer to a non-existing database!");
    }

    if(m_ctx->uuid.empty())
    {
        throw std::invalid_argument("Empty UUID provided!");
    }

    m_ctx->validator = std::make_shared<insert_validator>(m_ctx->registry);
    m_ctx->backend->initialize_schema();

    m_stmts = std::make_shared<data_storage::schema_v3::insert_statements>(m_ctx->backend,
                                                                           m_ctx->uuid);
    m_common_ops =
        std::make_shared<common_insert_operations<active_schema>>(m_ctx, m_stmts);
    m_info_writer = std::make_unique<info_registration_writer<active_schema>>(
        m_ctx, m_stmts, m_common_ops);
    m_kernel_dispatch_writer = std::make_unique<kernel_dispatch_writer<active_schema>>(
        m_ctx, m_stmts, m_common_ops);
    m_memory_copy_writer =
        std::make_unique<memory_copy_writer<active_schema>>(m_ctx, m_stmts, m_common_ops);
    m_memory_alloc_writer = std::make_unique<memory_alloc_writer<active_schema>>(
        m_ctx, m_stmts, m_common_ops);
    m_region_writer =
        std::make_unique<region_writer<active_schema>>(m_ctx, m_stmts, m_common_ops);
    m_pmc_event_writer =
        std::make_unique<pmc_event_writer<active_schema>>(m_ctx, m_stmts, m_common_ops);
}

}  // namespace rocstorage
