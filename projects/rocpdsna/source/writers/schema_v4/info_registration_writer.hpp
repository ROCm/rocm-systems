// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "writers/info_registration_writer.hpp"
#include "writers/schema_v4/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v4/insert_statements.hpp"
#include "data_storage/schema_version.hpp"

#include "common/string_conversions.hpp"
#include "debug.hpp"
#include "rocpdsna/writer_types.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace rocpdsna
{

namespace detail_v4
{

template <typename Utility, typename Entity>
[[nodiscard]] bool
is_already_registered(Utility& utility, const Entity& entity)
{
    if(utility.is_entry_registered(get_key(entity)))
    {
        LOG_WARNING("{} already registered", rocpdsna::to_string(entity));
        return true;
    }
    return false;
}

}  // namespace detail_v4

template <>
class info_registration_writer<data_storage::schema_v4_tag>
: public info_registration_writer_interface<
      info_registration_writer<data_storage::schema_v4_tag>>
{
public:
    explicit info_registration_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v4::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v4_tag>> common_ops)
    : m_ctx(std::move(ctx))
    , m_stmts(std::move(stmts))
    , m_common_ops(std::move(common_ops))
    {}

    void register_node_info_impl(const writer_types::node_info_t& node_info)
    {
        auto& node_info_utility = m_ctx->registry->node_info();
        if(detail_v4::is_already_registered(node_info_utility, node_info)) return;

        m_stmts->node_info_statement()(node_info.node_id,
                                       node_info.hash,
                                       node_info.machine_id,
                                       node_info.name,
                                       node_info.system_name,
                                       node_info.hostname,
                                       node_info.release,
                                       node_info.version,
                                       node_info.hardware_name,
                                       node_info.domain_name);

        node_info_utility.emplace_entity(node_info.node_id);
        LOG_TRACE("Registered node info: {}", rocpdsna::to_string(node_info));
    }

    void register_process_info_impl(const writer_types::process_info_t& process_info)
    {
        auto& process_info_utility = m_ctx->registry->process_info();
        if(detail_v4::is_already_registered(process_info_utility, process_info)) return;

        m_ctx->validator->require_node(process_info.node_id);

        const auto primary_key =
            m_ctx->key_providers->process_info().get_primary_key_value();

        m_stmts->process_info_statement()(primary_key,               // id
                                          process_info.node_id,      // nid
                                          process_info.ppid,         // ppid
                                          process_info.pid,          // pid
                                          process_info.name,         // name  v4
                                          process_info.init,         // init
                                          process_info.fini,         // fini
                                          process_info.start,        // start
                                          process_info.end,          // end
                                          process_info.command,      // command
                                          process_info.environment,  // environment
                                          process_info.extdata);     // extdata

        process_info_utility.emplace_entity(process_info.pid, primary_key);
        LOG_TRACE("Registered process info: {}", rocpdsna::to_string(process_info));
    }

    void register_agent_info_impl(const writer_types::agent_info_t& agent_info)
    {
        auto& agent_info_utility = m_ctx->registry->agent_info();
        if(detail_v4::is_already_registered(agent_info_utility, agent_info)) return;

        m_ctx->validator->require_node(agent_info.node_id)
            .require_process(agent_info.process_id);

        if(agent_info.unique_id.agent_type.has_value())
        {
            const std::string_view agent_type{ *agent_info.unique_id.agent_type };
            if(agent_type != "CPU" && agent_type != "GPU")
            {
                throw std::invalid_argument(
                    fmt::format("Invalid agent type: {}. Type can be NULL, CPU, or GPU.",
                                agent_type));
            }
        }

        const auto process_pk =
            m_ctx->validator->resolve_process_key(agent_info.process_id);
        const auto primary_key =
            m_ctx->key_providers->agent_info().get_primary_key_value();

        m_stmts->agent_info_statement()(
            primary_key,                      // id
            agent_info.node_id,               // nid
            process_pk,                       // pid
            agent_info.unique_id.agent_type,  // type
            agent_info.absolute_index,        // absolute_index
            agent_info.logical_index,         // logical_index
            agent_info.unique_id.type_index,  // type_index
            agent_info.uuid,                  // uuid
            agent_info.name,                  // name
            agent_info.generic_name,          // generic_name ← v4 only
            agent_info.model_name,            // model_name
            agent_info.vendor_name,           // vendor_name
            agent_info.product_name,          // product_name
            agent_info.extdata);              // extdata

        agent_info_utility.emplace_entity(agent_info.unique_id, primary_key);
        LOG_TRACE("Registered agent info: {}", rocpdsna::to_string(agent_info));
    }

    void register_pmc_info_impl(const writer_types::pmc_info_t& pmc_info)
    {
        auto& pmc_info_utility = m_ctx->registry->pmc_info();
        if(detail_v4::is_already_registered(pmc_info_utility, pmc_info)) return;

        m_ctx->validator->require_node(pmc_info.node_id)
            .require_process(pmc_info.process_id);

        // Only require agent if agent_id is provided (it's optional in the schema)
        if(pmc_info.unique_id.agent_id.has_value())
        {
            m_ctx->validator->require_agent(*pmc_info.unique_id.agent_id);
        }

        const auto process_pk =
            m_ctx->validator->resolve_process_key(pmc_info.process_id);

        // Resolve agent_pk only if agent_id has a value, otherwise use nullopt
        std::optional<primary_key_t> agent_pk = std::nullopt;
        if(pmc_info.unique_id.agent_id.has_value())
        {
            agent_pk = m_ctx->validator->resolve_agent_key(*pmc_info.unique_id.agent_id);
        }

        const auto primary_key = m_ctx->key_providers->pmc_info().get_primary_key_value();

        m_stmts->pmc_info_statement()(primary_key,                // id
                                      pmc_info.node_id,           // nid
                                      process_pk,                 // pid
                                      agent_pk,                   // agent_id
                                      pmc_info.target_arch,       // target_arch
                                      pmc_info.event_code,        // event_code
                                      pmc_info.instance_id,       // instance_id
                                      pmc_info.unique_id.name,    // name
                                      pmc_info.symbol,            // symbol
                                      pmc_info.qualifier,         // qualifier ← V4 ONLY
                                      pmc_info.description,       // description
                                      pmc_info.long_description,  // long_description
                                      pmc_info.component,         // component
                                      pmc_info.units,             // units
                                      pmc_info.value_type,        // value_type
                                      pmc_info.block,             // block
                                      pmc_info.expression,        // expression
                                      pmc_info.is_constant,       // is_constant
                                      pmc_info.is_derived,        // is_derived
                                      pmc_info.extdata);          // extdata

        pmc_info_utility.emplace_entity(pmc_info.unique_id, primary_key);
        LOG_TRACE("Registered pmc info: {}", rocpdsna::to_string(pmc_info));
    }

    void register_thread_info_impl(const writer_types::thread_info_t& thread_info)
    {
        auto& thread_info_utility = m_ctx->registry->thread_info();
        if(detail_v4::is_already_registered(thread_info_utility, thread_info)) return;

        m_ctx->validator->require_node(thread_info.node_id)
            .require_process(thread_info.process_id);

        const auto process_pk =
            m_ctx->validator->resolve_process_key(thread_info.process_id);
        const auto primary_key =
            m_ctx->key_providers->thread_info().get_primary_key_value();

        m_stmts->thread_info_statement()(primary_key,                    // id
                                         thread_info.node_id,            // nid
                                         thread_info.parent_process_id,  // ppid
                                         process_pk,             // pid (resolved FK)
                                         thread_info.thread_id,  // tid
                                         thread_info.name,       // name
                                         thread_info.start,      // start
                                         thread_info.end,        // end
                                         thread_info.extdata);   // extdata

        thread_info_utility.emplace_entity(thread_info.thread_id, primary_key);
        LOG_TRACE("Registered thread info: {}", rocpdsna::to_string(thread_info));
    }

    void register_stream_info_impl(const writer_types::stream_info_t& stream_info)
    {
        auto& stream_info_utility = m_ctx->registry->stream_info();
        if(detail_v4::is_already_registered(stream_info_utility, stream_info)) return;

        m_ctx->validator->require_node(stream_info.node_id)
            .require_process(stream_info.process_id);

        const auto process_pk =
            m_ctx->validator->resolve_process_key(stream_info.process_id);
        const auto primary_key =
            m_ctx->key_providers->stream_info().get_primary_key_value();

        m_stmts->stream_info_statement()(primary_key,
                                         stream_info.node_id,
                                         process_pk,
                                         stream_info.name,
                                         stream_info.extdata);

        stream_info_utility.emplace_entity(stream_info.stream_id, primary_key);
        LOG_TRACE("Registered stream info: {}", rocpdsna::to_string(stream_info));
    }

    void register_queue_info_impl(const writer_types::queue_info_t& queue_info)
    {
        auto& queue_info_utility = m_ctx->registry->queue_info();
        if(detail_v4::is_already_registered(queue_info_utility, queue_info)) return;

        m_ctx->validator->require_node(queue_info.node_id)
            .require_process(queue_info.process_id);

        const auto process_pk =
            m_ctx->validator->resolve_process_key(queue_info.process_id);
        const auto primary_key =
            m_ctx->key_providers->queue_info().get_primary_key_value();

        m_stmts->queue_info_statement()(primary_key,
                                        queue_info.node_id,
                                        process_pk,
                                        queue_info.name,
                                        queue_info.extdata);

        queue_info_utility.emplace_entity(queue_info.queue_id, primary_key);
        LOG_TRACE("Registered queue info: {}", rocpdsna::to_string(queue_info));
    }

    void register_code_object_info_impl(
        const writer_types::code_object_info_t& code_object_info)
    {
        auto& code_object_info_utility = m_ctx->registry->code_object_info();
        if(detail_v4::is_already_registered(code_object_info_utility, code_object_info))
            return;

        m_ctx->validator->require_node(code_object_info.node_id)
            .require_process(code_object_info.process_id)
            .validate_optional_agent(code_object_info.agent_id);

        const auto process_pk =
            m_ctx->validator->resolve_process_key(code_object_info.process_id);
        const auto agent_pk =
            m_ctx->validator->resolve_optional_agent_key(code_object_info.agent_id);

        m_stmts->code_object_info_statement()(code_object_info.id,
                                              code_object_info.node_id,
                                              process_pk,
                                              agent_pk,
                                              code_object_info.uri,
                                              code_object_info.load_base,
                                              code_object_info.load_size,
                                              code_object_info.load_delta,
                                              code_object_info.storage_type,
                                              code_object_info.extdata);

        code_object_info_utility.emplace_entity(code_object_info.id);
        LOG_TRACE("Registered code object info: {}",
                  rocpdsna::to_string(code_object_info));
    }

    void register_kernel_symbol_info_impl(
        const writer_types::kernel_symbol_info_t& kernel_symbol_info)
    {
        auto& kernel_symbol_info_utility = m_ctx->registry->kernel_symbol_info();
        if(detail_v4::is_already_registered(kernel_symbol_info_utility,
                                            kernel_symbol_info))
            return;

        m_ctx->validator->require_node(kernel_symbol_info.node_id)
            .require_process(kernel_symbol_info.process_id)
            .require_code_object(kernel_symbol_info.code_obj_id);

        const auto process_pk =
            m_ctx->validator->resolve_process_key(kernel_symbol_info.process_id);

        m_stmts->kernel_symbol_info_statement()(
            kernel_symbol_info.id,
            kernel_symbol_info.node_id,
            process_pk,
            kernel_symbol_info.code_obj_id,
            kernel_symbol_info.name,
            kernel_symbol_info.display_name,
            kernel_symbol_info.kernel_object,
            kernel_symbol_info.kernarg_segment_size,
            kernel_symbol_info.kernarg_segment_alignment,
            kernel_symbol_info.group_segment_size,
            kernel_symbol_info.private_segment_size,
            kernel_symbol_info.sgpr_count,
            kernel_symbol_info.arch_vgpr_count,
            kernel_symbol_info.accum_vgpr_count,
            kernel_symbol_info.extdata);

        kernel_symbol_info_utility.emplace_entity(kernel_symbol_info.id);
        LOG_TRACE("Registered kernel symbol info: {}",
                  rocpdsna::to_string(kernel_symbol_info));
    }

    void register_track_info_impl(const writer_types::track_info_t& track)
    {
        auto& track_info_utility = m_ctx->registry->track_info();
        if(detail_v4::is_already_registered(track_info_utility, track)) return;

        m_ctx->validator->require_node(track.node_id)
            .validate_optional_process(track.process_id)
            .validate_optional_thread(track.thread_id)
            .validate_optional_agent(track.agent_id)
            .validate_optional_queue(track.queue_id)
            .validate_optional_stream(track.stream_id);

        if(track.name.has_value() &&
           !m_ctx->registry->string_info().is_entry_registered(track.name.value()))
        {
            m_common_ops->register_string(track.name.value());
        }

        const auto process_pk =
            m_ctx->validator->resolve_optional_process_key(track.process_id);
        const auto thread_pk =
            m_ctx->validator->resolve_optional_thread_key(track.thread_id);
        const auto agent_pk =
            m_ctx->validator->resolve_optional_agent_key(track.agent_id);
        const auto queue_pk =
            m_ctx->validator->resolve_optional_queue_key(track.queue_id);
        const auto stream_pk =
            m_ctx->validator->resolve_optional_stream_key(track.stream_id);
        const auto string_pk = m_ctx->validator->resolve_optional_string_key(
            track.name.has_value() ? std::make_optional<std::string>(track.name.value())
                                   : std::nullopt);
        const auto primary_key =
            m_ctx->key_providers->track_info().get_primary_key_value();

        std::optional<primary_key_t> ppid_fk = std::nullopt;
        if(track.ppid.has_value())
        {
            ppid_fk = static_cast<primary_key_t>(track.ppid.value());
        }

        m_stmts->track_info_statement()(primary_key,
                                        track.node_id,
                                        ppid_fk,
                                        process_pk,
                                        thread_pk,
                                        agent_pk,
                                        queue_pk,
                                        stream_pk,
                                        string_pk,
                                        track.extdata);

        track_info_utility.emplace_entity(track, primary_key);
        LOG_TRACE("Registered track info: {}", rocpdsna::to_string(track));
    }

    /**
     * @brief Register category info (v4+ table)
     * @note Uses m_common_ops->get_or_insert_category which handles caching
     */
    void register_category_info_impl(const writer_types::category_info_t& category_info)
    {
        m_common_ops->ensure_category_registered(category_info.name,
                                                 category_info.extdata);
        LOG_TRACE("Registered category info: id={}, name={}",
                  category_info.id,
                  category_info.name);
    }

    /**
     * @brief Register address range info (v4+ table)
     */
    void register_address_range_info_impl(
        const writer_types::address_range_info_t& addr_range)
    {
        auto& addr_range_utility = m_ctx->registry->address_range_info();
        if(detail_v4::is_already_registered(addr_range_utility, addr_range)) return;

        m_ctx->validator->require_node(addr_range.node_id);
        m_ctx->validator->require_process(addr_range.process_id);

        const auto process_pk =
            m_ctx->validator->resolve_process_key(addr_range.process_id);
        const auto primary_key =
            m_ctx->key_providers->address_range_info().get_primary_key_value();

        m_stmts->address_range_info_statement()(primary_key,
                                                addr_range.node_id,
                                                process_pk,
                                                addr_range.address_base,
                                                addr_range.address_low,
                                                addr_range.address_high,
                                                addr_range.extdata);

        addr_range_utility.emplace_entity(addr_range.id, primary_key);
        LOG_TRACE(
            "Registered address range info: id={}, base={:#x}, low={:#x}, high={:#x}",
            addr_range.id,
            addr_range.address_base,
            addr_range.address_low,
            addr_range.address_high);
    }

    /**
     * @brief Register source code info (v4+ table)
     */
    void register_source_code_info_impl(
        const writer_types::source_code_info_t& source_code)
    {
        auto& source_code_utility = m_ctx->registry->source_code_info();
        if(detail_v4::is_already_registered(source_code_utility, source_code)) return;

        m_ctx->validator->require_node(source_code.node_id);
        m_ctx->validator->require_process(source_code.process_id);

        const auto process_pk =
            m_ctx->validator->resolve_process_key(source_code.process_id);

        std::optional<primary_key_t> address_pk = std::nullopt;
        if(source_code.address_id.has_value())
        {
            const auto addr_id = source_code.address_id.value();
            if(!m_ctx->registry->address_range_info().is_entry_registered(addr_id))
            {
                throw std::invalid_argument(
                    fmt::format("source_code_info references address_range_id {} "
                                "which has not been registered. "
                                "Call register_address_range_info() first.",
                                addr_id));
            }
            address_pk =
                m_ctx->registry->address_range_info().get_primary_key_value_for_entity(
                    addr_id);
        }

        const auto primary_key =
            m_ctx->key_providers->source_code_info().get_primary_key_value();

        m_stmts->source_code_info_statement()(primary_key,
                                              source_code.node_id,
                                              process_pk,
                                              address_pk,
                                              source_code.file,
                                              source_code.line_number,
                                              source_code.lines,
                                              source_code.instructions,
                                              source_code.extdata);

        source_code_utility.emplace_entity(source_code.id, primary_key);
        LOG_TRACE("Registered source code info: {}", rocpdsna::to_string(source_code));
    }

    /**
     * @brief Register program counter info (v4+ table)
     */
    void register_pc_info_impl(const writer_types::pc_info_t& pc_info)
    {
        auto& pc_info_utility = m_ctx->registry->pc_info();
        if(detail_v4::is_already_registered(pc_info_utility, pc_info)) return;

        m_ctx->validator->require_node(pc_info.node_id);
        m_ctx->validator->require_process(pc_info.process_id);

        const auto process_pk = m_ctx->validator->resolve_process_key(pc_info.process_id);

        std::optional<primary_key_t> address_pk = std::nullopt;
        if(pc_info.address_id.has_value())
        {
            const auto addr_id = pc_info.address_id.value();
            if(!m_ctx->registry->address_range_info().is_entry_registered(addr_id))
            {
                throw std::invalid_argument(
                    fmt::format("pc_info references address_range_id {} "
                                "which has not been registered. "
                                "Call register_address_range_info() first.",
                                addr_id));
            }
            address_pk =
                m_ctx->registry->address_range_info().get_primary_key_value_for_entity(
                    addr_id);
        }

        const auto primary_key = m_ctx->key_providers->pc_info().get_primary_key_value();

        m_stmts->pc_info_statement()(primary_key,
                                     pc_info.node_id,
                                     process_pk,
                                     pc_info.function,
                                     address_pk,
                                     pc_info.file,
                                     pc_info.line,
                                     pc_info.extdata);

        pc_info_utility.emplace_entity(pc_info.id, primary_key);
        LOG_TRACE("Registered pc info: {}", rocpdsna::to_string(pc_info));
    }

    void register_string_impl(std::string_view str)
    {
        m_common_ops->register_string(str);
    }

private:
    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v4::insert_statements<data_storage::sqlite_backend>>
                                                                           m_stmts;
    std::shared_ptr<common_insert_operations<data_storage::schema_v4_tag>> m_common_ops;
};

}  // namespace rocpdsna
