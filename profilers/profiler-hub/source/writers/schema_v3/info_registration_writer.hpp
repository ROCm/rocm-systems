// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "writers/info_registration_writer.hpp"
#include "writers/schema_v3/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v3/insert_statements.hpp"
#include "data_storage/schema_version.hpp"

#include "common/string_conversions.hpp"
#include "debug.hpp"
#include "profiler-hub/writer_types.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace profiler_hub
{

namespace
{

template <typename Utility, typename Entity>
[[nodiscard]] bool
is_already_registered(Utility& utility, const Entity& entity)
{
    if(utility.is_entry_registered(get_key(entity)))
    {
        LOG_WARNING("{} already registered", profiler_hub::to_string(entity));
        return true;
    }
    return false;
}

}  // namespace

template <>
class info_registration_writer<data_storage::schema_v3_tag>
: public info_registration_writer_interface<
      info_registration_writer<data_storage::schema_v3_tag>>
{
public:
    explicit info_registration_writer(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                               stmts,
        std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> common_ops)
    : m_ctx(std::move(ctx))
    , m_stmts(std::move(stmts))
    , m_common_ops(std::move(common_ops))
    {}

    void register_node_info_impl(const writer_types::node_info_t& node_info)
    {
        auto& node_info_utility = m_ctx->registry->node_info();
        if(is_already_registered(node_info_utility, node_info)) return;

        m_stmts->node_info_statement()(node_info.node_id,
                                       node_info.hash,
                                       node_info.machine_id,
                                       node_info.system_name,
                                       node_info.hostname,
                                       node_info.release,
                                       node_info.version,
                                       node_info.hardware_name,
                                       node_info.domain_name);

        node_info_utility.emplace_entity(node_info.node_id);
        LOG_TRACE("Registered node info: {}", profiler_hub::to_string(node_info));
    }

    void register_process_info_impl(const writer_types::process_info_t& process_info)
    {
        auto& process_info_utility = m_ctx->registry->process_info();
        if(is_already_registered(process_info_utility, process_info)) return;

        m_ctx->validator->require_node(process_info.node_id);

        const auto primary_key =
            m_ctx->key_providers->process_info().get_primary_key_value();

        m_stmts->process_info_statement()(primary_key,
                                          process_info.node_id,
                                          process_info.ppid,
                                          process_info.pid,
                                          process_info.init,
                                          process_info.fini,
                                          process_info.start,
                                          process_info.end,
                                          process_info.command,
                                          process_info.environment,
                                          process_info.extdata);

        process_info_utility.emplace_entity(process_info.pid, primary_key);
        LOG_TRACE("Registered process info: {}", profiler_hub::to_string(process_info));
    }

    void register_agent_info_impl(const writer_types::agent_info_t& agent_info)
    {
        auto& agent_info_utility = m_ctx->registry->agent_info();
        if(is_already_registered(agent_info_utility, agent_info)) return;

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

        m_stmts->agent_info_statement()(primary_key,
                                        agent_info.node_id,
                                        process_pk,
                                        agent_info.unique_id.agent_type,
                                        agent_info.absolute_index,
                                        agent_info.logical_index,
                                        agent_info.unique_id.type_index,
                                        agent_info.uuid,
                                        agent_info.name,
                                        agent_info.model_name,
                                        agent_info.vendor_name,
                                        agent_info.product_name,
                                        agent_info.user_name,
                                        agent_info.extdata);

        agent_info_utility.emplace_entity(agent_info.unique_id, primary_key);
        LOG_TRACE("Registered agent info: {}", profiler_hub::to_string(agent_info));
    }

    void register_pmc_info_impl(const writer_types::pmc_info_t& pmc_info)
    {
        auto& pmc_info_utility = m_ctx->registry->pmc_info();
        if(is_already_registered(pmc_info_utility, pmc_info)) return;

        m_ctx->validator->require_node(pmc_info.node_id)
            .require_process(pmc_info.process_id)
            .validate_optional_agent(pmc_info.unique_id.agent_id);

        const auto process_pk =
            m_ctx->validator->resolve_process_key(pmc_info.process_id);
        const auto agent_pk =
            m_ctx->validator->resolve_optional_agent_key(pmc_info.unique_id.agent_id);
        const auto primary_key = m_ctx->key_providers->pmc_info().get_primary_key_value();

        m_stmts->pmc_info_statement()(primary_key,
                                      pmc_info.node_id,
                                      process_pk,
                                      agent_pk,
                                      pmc_info.target_arch,
                                      pmc_info.event_code,
                                      pmc_info.instance_id,
                                      pmc_info.unique_id.name,
                                      pmc_info.symbol,
                                      pmc_info.description,
                                      pmc_info.long_description,
                                      pmc_info.component,
                                      pmc_info.units,
                                      pmc_info.value_type,
                                      pmc_info.block,
                                      pmc_info.expression,
                                      pmc_info.is_constant,
                                      pmc_info.is_derived,
                                      pmc_info.extdata);

        pmc_info_utility.emplace_entity(pmc_info.unique_id, primary_key);
        LOG_TRACE("Registered pmc info: {}", profiler_hub::to_string(pmc_info));
    }

    void register_thread_info_impl(const writer_types::thread_info_t& thread_info)
    {
        auto& thread_info_utility = m_ctx->registry->thread_info();
        if(is_already_registered(thread_info_utility, thread_info)) return;

        m_ctx->validator->require_node(thread_info.node_id)
            .require_process(thread_info.process_id);

        const auto process_pk =
            m_ctx->validator->resolve_process_key(thread_info.process_id);
        const auto primary_key =
            m_ctx->key_providers->thread_info().get_primary_key_value();

        m_stmts->thread_info_statement()(primary_key,
                                         thread_info.node_id,
                                         thread_info.parent_process_id,
                                         process_pk,
                                         thread_info.thread_id,
                                         thread_info.name,
                                         thread_info.start,
                                         thread_info.end,
                                         thread_info.extdata);

        thread_info_utility.emplace_entity(thread_info.thread_id, primary_key);
        LOG_TRACE("Registered thread info: {}", profiler_hub::to_string(thread_info));
    }

    void register_stream_info_impl(const writer_types::stream_info_t& stream_info)
    {
        auto& stream_info_utility = m_ctx->registry->stream_info();
        if(is_already_registered(stream_info_utility, stream_info)) return;

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
        LOG_TRACE("Registered stream info: {}", profiler_hub::to_string(stream_info));
    }

    void register_queue_info_impl(const writer_types::queue_info_t& queue_info)
    {
        auto& queue_info_utility = m_ctx->registry->queue_info();
        if(is_already_registered(queue_info_utility, queue_info)) return;

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
        LOG_TRACE("Registered queue info: {}", profiler_hub::to_string(queue_info));
    }

    void register_code_object_info_impl(
        const writer_types::code_object_info_t& code_object_info)
    {
        auto& code_object_info_utility = m_ctx->registry->code_object_info();
        if(is_already_registered(code_object_info_utility, code_object_info)) return;

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
                  profiler_hub::to_string(code_object_info));
    }

    void register_kernel_symbol_info_impl(
        const writer_types::kernel_symbol_info_t& kernel_symbol_info)
    {
        auto& kernel_symbol_info_utility = m_ctx->registry->kernel_symbol_info();
        if(is_already_registered(kernel_symbol_info_utility, kernel_symbol_info)) return;

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
                  profiler_hub::to_string(kernel_symbol_info));
    }

    void register_track_info_impl(const writer_types::track_info_t& track)
    {
        auto& track_info_utility = m_ctx->registry->track_info();
        if(is_already_registered(track_info_utility, track)) return;

        m_ctx->validator->require_node(track.node_id)
            .validate_optional_process(track.process_id)
            .validate_optional_thread(track.thread_id);

        if(track.name.has_value() &&
           !m_ctx->registry->string_info().is_entry_registered(track.name.value()))
        {
            m_common_ops->register_string(track.name.value());
        }

        const auto process_pk =
            m_ctx->validator->resolve_optional_process_key(track.process_id);
        const auto thread_pk =
            m_ctx->validator->resolve_optional_thread_key(track.thread_id);
        const auto string_pk = m_ctx->validator->resolve_optional_string_key(
            track.name.has_value() ? std::make_optional<std::string>(track.name.value())
                                   : std::nullopt);
        const auto primary_key =
            m_ctx->key_providers->track_info().get_primary_key_value();

        m_stmts->track_info_statement()(
            primary_key, track.node_id, process_pk, thread_pk, string_pk, track.extdata);

        track_info_utility.emplace_entity(track, primary_key);
        LOG_TRACE("Registered track info: {}", profiler_hub::to_string(track));
    }

    void register_string_impl(std::string_view str)
    {
        m_common_ops->register_string(str);
    }

private:
    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
                                                                           m_stmts;
    std::shared_ptr<common_insert_operations<data_storage::schema_v3_tag>> m_common_ops;
};

}  // namespace profiler_hub
