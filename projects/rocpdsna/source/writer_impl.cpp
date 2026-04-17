// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "writer_impl.hpp"
#include "storage_impl.hpp"

#include <memory>
#include <stdexcept>

namespace rocpdsna
{

template <typename Policy>
writer_impl_core<Policy>::writer_impl_core(std::shared_ptr<writer_context> ctx)
: m_ctx(std::move(ctx))
, m_stmts(std::make_shared<stmts_t>(m_ctx->backend, m_ctx->uuid))
, m_common_ops(std::make_shared<common_ops_t>(m_ctx, m_stmts))
, m_info_writer(
      std::make_unique<typename Policy::info_writer_t>(m_ctx, m_stmts, m_common_ops))
, m_kernel_dispatch_writer(
      std::make_unique<typename Policy::kernel_dispatch_writer_t>(m_ctx,
                                                                  m_stmts,
                                                                  m_common_ops))
, m_memory_copy_writer(
      std::make_unique<typename Policy::memory_copy_writer_t>(m_ctx,
                                                              m_stmts,
                                                              m_common_ops))
, m_memory_alloc_writer(
      std::make_unique<typename Policy::memory_alloc_writer_t>(m_ctx,
                                                               m_stmts,
                                                               m_common_ops))
, m_region_writer(
      std::make_unique<typename Policy::region_writer_t>(m_ctx, m_stmts, m_common_ops))
, m_pmc_event_writer(
      std::make_unique<typename Policy::pmc_event_writer_t>(m_ctx, m_stmts, m_common_ops))
{}

// Explicitly instantiate both v3 and v4 implementations (always compiled)
template class writer_impl_core<writer_policy_v3>;
template class writer_impl_core<writer_policy_v4>;
template class writer_impl_polymorphic<writer_policy_v3>;
template class writer_impl_polymorphic<writer_policy_v4>;

std::shared_ptr<writer_context>
writer_t::impl::create_writer_context(storage_t& storage, version_t version)
{
    auto ctx = std::make_shared<writer_context>();
    ctx->backend =
        storage.m_impl->create_database(storage_t::impl::storage_type_t::write);
    ctx->registry      = std::make_shared<entity_registry>();
    ctx->key_providers = std::make_shared<primary_key_providers>();
    ctx->uuid          = storage.m_impl->get_uuid();

    if(!ctx->backend)
    {
        throw std::invalid_argument("Provided pointer to a non-existing database!");
    }

    if(ctx->uuid.empty())
    {
        throw std::invalid_argument("Empty UUID provided!");
    }

    ctx->validator = std::make_shared<insert_validator>(ctx->registry);
    ctx->backend->initialize_schema(version);

    return ctx;
}

/**
 * @brief Constructor with runtime schema selection
 *
 * Queries the storage instance for its configured schema version
 * and instantiates the appropriate writer implementation (v3 or v4).
 * Supports LATEST marker which resolves to the most recent schema.
 */
writer_t::impl::impl(std::unique_ptr<rocpdsna::storage_t> storage)
: m_storage(std::move(storage))
, m_version(m_storage->get_storage_version())
{
    // Resolve LATEST marker to actual version
    auto actual_version = m_version;

    // Create writer context with schema version for proper initialization
    auto ctx = create_writer_context(*m_storage, actual_version);

    // Runtime dispatch based on schema version
    if(actual_version == rocpdsna::version_t{ 4, 0, 0 })
    {
        m_impl = std::make_unique<writer_impl_polymorphic<writer_policy_v4>>(ctx);
    }
    else if(actual_version == rocpdsna::version_t{ 3, 0, 0 })
    {
        // Default to v3 (backward compatible)
        m_impl = std::make_unique<writer_impl_polymorphic<writer_policy_v3>>(ctx);
    }
}

}  // namespace rocpdsna
