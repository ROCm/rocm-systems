// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "writer_impl.hpp"
#include "storage_impl.hpp"

#include "profiler-hub/version.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

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
, m_graph_launch_writer(
      std::make_unique<typename Policy::graph_launch_writer_t>(m_ctx,
                                                               m_stmts,
                                                               m_common_ops))
, m_hip_event_writer(
      std::make_unique<typename Policy::hip_event_writer_t>(m_ctx, m_stmts, m_common_ops))
{}

template class writer_impl_core<writer_policy_3_0_0>;
template class writer_impl_core<writer_policy_3_0_1>;
template class writer_impl_core<writer_policy_3_0_2>;
template class writer_impl_core<writer_policy_3_0_3>;
template class writer_impl_core<writer_policy_3_0_4>;
template class writer_impl_core<writer_policy_3_0_5>;

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

    ctx->validator      = std::make_shared<insert_validator>(ctx->registry);
    ctx->schema_version = storage->get_storage_version();

    ctx->backend->initialize_schema(ctx->schema_version);

    return ctx;
}

writer_variant_t
writer_t::impl::create_core(const std::unique_ptr<storage_t>& storage)
{
    if(!storage)
    {
        throw std::invalid_argument("Provided pointer to a non-existing storage!");
    }

    const auto version = storage->get_storage_version();
    auto       ctx     = create_writer_context(storage);

    if(version == version_t{ 3, 0, 0 })
    {
        return writer_impl_core<writer_policy_3_0_0>(std::move(ctx));
    }
    if(version == version_t{ 3, 0, 1 })
    {
        return writer_impl_core<writer_policy_3_0_1>(std::move(ctx));
    }
    if(version == version_t{ 3, 0, 2 })
    {
        return writer_impl_core<writer_policy_3_0_2>(std::move(ctx));
    }
    if(version == version_t{ 3, 0, 3 })
    {
        return writer_impl_core<writer_policy_3_0_3>(std::move(ctx));
    }
    if(version == version_t{ 3, 0, 4 })
    {
        return writer_impl_core<writer_policy_3_0_4>(std::move(ctx));
    }
    if(version == version_t{ 3, 0, 5 })
    {
        return writer_impl_core<writer_policy_3_0_5>(std::move(ctx));
    }

    throw std::runtime_error("No writer for schema " + version.to_string());
}

writer_t::impl::impl(std::unique_ptr<profiler_hub::storage_t> storage)
: m_core(create_core(storage))
, m_storage(std::move(storage))
{}

}  // namespace profiler_hub
