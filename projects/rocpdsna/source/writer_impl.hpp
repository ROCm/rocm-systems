// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "rocpdsna/storage.hpp"
#include "rocpdsna/writer.hpp"
#include "rocpdsna/writer_types.hpp"

#include "data_storage/schema_version.hpp"
#include "rocpdsna/storage_types.hpp"
#include "writers/writer_context.hpp"
#include "writers/writer_policy_traits.hpp"

// Always include both schema versions for runtime selection
#include "writers/schema_v3/writer_policy.hpp"
#include "writers/schema_v4/writer_policy.hpp"

#include <memory>
#include <variant>

namespace rocpdsna
{

template <typename Policy>
class writer_impl_core
{
    static_assert(is_valid_writer_policy_v<Policy>,
                  "Policy must satisfy writer policy requirements: "
                  "provide all required type aliases and writer types "
                  "must inherit from their CRTP interfaces");

    using stmts_t      = typename Policy::insert_statements_t;
    using common_ops_t = typename Policy::common_ops_t;

public:
    explicit writer_impl_core(std::shared_ptr<writer_context> ctx);

    void register_node_info(const writer_types::node_info_t& node_info)
    {
        m_info_writer->register_node_info(node_info);
    }

    void register_process_info(const writer_types::process_info_t& process_info)
    {
        m_info_writer->register_process_info(process_info);
    }

    void register_agent_info(const writer_types::agent_info_t& agent_info)
    {
        m_info_writer->register_agent_info(agent_info);
    }

    void register_pmc_info(const writer_types::pmc_info_t& pmc_info)
    {
        m_info_writer->register_pmc_info(pmc_info);
    }

    void register_thread_info(const writer_types::thread_info_t& thread_info)
    {
        m_info_writer->register_thread_info(thread_info);
    }

    void register_stream_info(const writer_types::stream_info_t& stream_info)
    {
        m_info_writer->register_stream_info(stream_info);
    }

    void register_queue_info(const writer_types::queue_info_t& queue_info)
    {
        m_info_writer->register_queue_info(queue_info);
    }

    void register_code_object_info(
        const writer_types::code_object_info_t& code_object_info)
    {
        m_info_writer->register_code_object_info(code_object_info);
    }

    void register_kernel_symbol_info(
        const writer_types::kernel_symbol_info_t& kernel_symbol_info)
    {
        m_info_writer->register_kernel_symbol_info(kernel_symbol_info);
    }

    void register_track_info(const writer_types::track_info_t& track)
    {
        m_info_writer->register_track_info(track);
    }

    void register_string(std::string_view str) { m_info_writer->register_string(str); }

    void register_category_info(const writer_types::category_info_t& category_info)
    {
        m_info_writer->register_category_info(category_info);
    }

    void register_address_range_info(
        const writer_types::address_range_info_t& address_range)
    {
        m_info_writer->register_address_range_info(address_range);
    }

    void register_source_code_info(const writer_types::source_code_info_t& source_code)
    {
        m_info_writer->register_source_code_info(source_code);
    }

    void register_pc_info(const writer_types::pc_info_t& pc_info)
    {
        m_info_writer->register_pc_info(pc_info);
    }

    void insert_region_data(const writer_types::region_data_t&       region_data,
                            const writer_types::trace_environment_t& trace_environment)
    {
        m_region_writer->insert(region_data, trace_environment);
    }

    void insert_pmc_event_data(const writer_types::pmc_event_data_t&     pmc_event_data,
                               const writer_types::pmc_info_unique_id_t& pmc_unique_id)
    {
        m_pmc_event_writer->insert(pmc_event_data, pmc_unique_id);
    }

    void insert_kernel_dispatch_data(
        const writer_types::kernel_dispatch_data_t& kernel_dispatch_data,
        const writer_types::trace_environment_t&    trace_environment)
    {
        m_kernel_dispatch_writer->insert(kernel_dispatch_data, trace_environment);
    }

    void insert_memory_copy_data(
        const writer_types::memory_copy_data_t&  memory_copy_data,
        const writer_types::trace_environment_t& trace_environment)
    {
        m_memory_copy_writer->insert(memory_copy_data, trace_environment);
    }

    void insert_memory_alloc_data(
        const writer_types::memory_alloc_data_t& memory_alloc_data,
        const writer_types::trace_environment_t& trace_environment)
    {
        m_memory_alloc_writer->insert(memory_alloc_data, trace_environment);
    }

    void flush_in_memory_data_to_disk() { m_ctx->backend->flush(); }

    void set_sqlite_foreign_keys_enabled(bool enabled)
    {
        m_ctx->backend->set_foreign_keys_enabled(enabled);
    }

private:
    std::shared_ptr<writer_context>                            m_ctx;
    std::shared_ptr<stmts_t>                                   m_stmts;
    std::shared_ptr<common_ops_t>                              m_common_ops;
    std::unique_ptr<typename Policy::info_writer_t>            m_info_writer;
    std::unique_ptr<typename Policy::kernel_dispatch_writer_t> m_kernel_dispatch_writer;
    std::unique_ptr<typename Policy::memory_copy_writer_t>     m_memory_copy_writer;
    std::unique_ptr<typename Policy::memory_alloc_writer_t>    m_memory_alloc_writer;
    std::unique_ptr<typename Policy::region_writer_t>          m_region_writer;
    std::unique_ptr<typename Policy::pmc_event_writer_t>       m_pmc_event_writer;
};

using writer_variant_t =
    std::variant<writer_impl_core<writer_policy_v3>, writer_impl_core<writer_policy_v4>>;

/**
 * @brief Writer implementation with runtime schema selection via std::variant
 */
struct writer_t::impl
{
    explicit impl(std::unique_ptr<rocpdsna::storage_t> storage);

    void register_node_info(const writer_types::node_info_t& v)
    {
        dispatch([&](auto& c) { c.register_node_info(v); });
    }
    void register_process_info(const writer_types::process_info_t& v)
    {
        dispatch([&](auto& c) { c.register_process_info(v); });
    }
    void register_agent_info(const writer_types::agent_info_t& v)
    {
        dispatch([&](auto& c) { c.register_agent_info(v); });
    }
    void register_pmc_info(const writer_types::pmc_info_t& v)
    {
        dispatch([&](auto& c) { c.register_pmc_info(v); });
    }
    void register_thread_info(const writer_types::thread_info_t& v)
    {
        dispatch([&](auto& c) { c.register_thread_info(v); });
    }
    void register_stream_info(const writer_types::stream_info_t& v)
    {
        dispatch([&](auto& c) { c.register_stream_info(v); });
    }
    void register_queue_info(const writer_types::queue_info_t& v)
    {
        dispatch([&](auto& c) { c.register_queue_info(v); });
    }
    void register_code_object_info(const writer_types::code_object_info_t& v)
    {
        dispatch([&](auto& c) { c.register_code_object_info(v); });
    }
    void register_kernel_symbol_info(const writer_types::kernel_symbol_info_t& v)
    {
        dispatch([&](auto& c) { c.register_kernel_symbol_info(v); });
    }
    void register_track_info(const writer_types::track_info_t& v)
    {
        dispatch([&](auto& c) { c.register_track_info(v); });
    }
    void register_string(std::string_view v)
    {
        dispatch([&](auto& c) { c.register_string(v); });
    }
    void register_category_info(const writer_types::category_info_t& v)
    {
        dispatch([&](auto& c) { c.register_category_info(v); });
    }
    void register_address_range_info(const writer_types::address_range_info_t& v)
    {
        dispatch([&](auto& c) { c.register_address_range_info(v); });
    }
    void register_source_code_info(const writer_types::source_code_info_t& v)
    {
        dispatch([&](auto& c) { c.register_source_code_info(v); });
    }
    void register_pc_info(const writer_types::pc_info_t& v)
    {
        dispatch([&](auto& c) { c.register_pc_info(v); });
    }
    void insert_region_data(const writer_types::region_data_t&       a,
                            const writer_types::trace_environment_t& b)
    {
        dispatch([&](auto& c) { c.insert_region_data(a, b); });
    }
    void insert_pmc_event_data(const writer_types::pmc_event_data_t&     a,
                               const writer_types::pmc_info_unique_id_t& b)
    {
        dispatch([&](auto& c) { c.insert_pmc_event_data(a, b); });
    }
    void insert_kernel_dispatch_data(const writer_types::kernel_dispatch_data_t& a,
                                     const writer_types::trace_environment_t&    b)
    {
        dispatch([&](auto& c) { c.insert_kernel_dispatch_data(a, b); });
    }
    void insert_memory_copy_data(const writer_types::memory_copy_data_t&  a,
                                 const writer_types::trace_environment_t& b)
    {
        dispatch([&](auto& c) { c.insert_memory_copy_data(a, b); });
    }
    void insert_memory_alloc_data(const writer_types::memory_alloc_data_t& a,
                                  const writer_types::trace_environment_t& b)
    {
        dispatch([&](auto& c) { c.insert_memory_alloc_data(a, b); });
    }
    void flush_in_memory_data_to_disk()
    {
        dispatch([](auto& c) { c.flush_in_memory_data_to_disk(); });
    }
    void set_sqlite_foreign_keys_enabled(bool enabled)
    {
        dispatch([&](auto& c) { c.set_sqlite_foreign_keys_enabled(enabled); });
    }

private:
    template <typename F>
    void dispatch(F&& f)
    {
        std::visit(std::forward<F>(f), m_core);
    }

    static std::shared_ptr<writer_context> create_writer_context(storage_t& storage,
                                                                 version_t  version);

    static writer_variant_t create_core(storage_t& storage, version_t version);

    std::unique_ptr<rocpdsna::storage_t> m_storage;
    version_t                            m_version;
    writer_variant_t                     m_core;
};

}  // namespace rocpdsna
