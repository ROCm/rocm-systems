// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "writer_impl.hpp"
#include "storage_impl.hpp"

#include <memory>
#include <stdexcept>

namespace profiler_hub
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

template class writer_impl_core<active_policy_t>;

std::shared_ptr<writer_context>
writer_t::impl::create_writer_context(const std::unique_ptr<storage_t>& storage)
{
    if(!storage)
    {
        throw std::invalid_argument("Provided pointer to a non-existing storage!");
    }

    auto ctx = std::make_shared<writer_context>();
    ctx->backend =
        storage->m_impl->create_database(storage_t::impl::storage_type_t::write);
    ctx->registry      = std::make_shared<entity_registry>();
    ctx->key_providers = std::make_shared<primary_key_providers>();
    ctx->uuid          = storage->m_impl->get_uuid();

    if(!ctx->backend)
    {
        throw std::invalid_argument("Provided pointer to a non-existing database!");
    }

    if(ctx->uuid.empty())
    {
        throw std::invalid_argument("Empty UUID provided!");
    }

    ctx->validator = std::make_shared<insert_validator>(ctx->registry);
    ctx->backend->initialize_schema();

    return ctx;
}

writer_t::impl::impl(std::unique_ptr<profiler_hub::storage_t> storage)
: writer_impl_core<active_policy_t>(create_writer_context(storage))
, m_storage(std::move(storage))
{}

}  // namespace profiler_hub
