// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <rocstorage/writer_types.hpp>

#include "data_indentifier.hpp"
#include "string_conversions.hpp"

#include <memory>
#include <stdexcept>

namespace rocstorage
{

class insert_validator
{
public:
    explicit insert_validator(const std::shared_ptr<data_identifiers>& identifiers)
    : m_identifiers(std::move(identifiers))
    {}

    insert_validator& require_node(const std::optional<writer_api::node_id_t>& node_id)
    {
        validate_required(m_identifiers->node_info(), node_id, "Node", "node_id");
        return *this;
    }

    insert_validator& require_node(writer_api::node_id_t node_id)
    {
        validate_direct(m_identifiers->node_info(), node_id, "Node", "node_id");
        return *this;
    }

    insert_validator& require_process(
        const std::optional<writer_api::process_id_t>& process_id)
    {
        validate_required(m_identifiers->process_info(), process_id, "Process", "pid");
        return *this;
    }

    insert_validator& require_process(writer_api::process_id_t process_id)
    {
        validate_direct(m_identifiers->process_info(), process_id, "Process", "pid");
        return *this;
    }

    insert_validator& require_thread(
        const std::optional<writer_api::thread_id_t>& thread_id)
    {
        validate_required(m_identifiers->thread_info(), thread_id, "Thread", "thread_id");
        return *this;
    }

    insert_validator& require_agent(
        const std::optional<writer_api::agent_unique_id_t>& agent_id)
    {
        validate_required(m_identifiers->agent_info(), agent_id, "Agent", "agent_id");
        return *this;
    }

    insert_validator& require_agent(const writer_api::agent_unique_id_t& agent_id)
    {
        validate_direct(m_identifiers->agent_info(), agent_id, "Agent", "agent_id");
        return *this;
    }

    insert_validator& require_queue(const std::optional<writer_api::queue_id_t>& queue_id)
    {
        validate_required(m_identifiers->queue_info(), queue_id, "Queue", "queue_id");
        return *this;
    }

    insert_validator& require_stream(
        const std::optional<writer_api::stream_id_t>& stream_id)
    {
        validate_required(m_identifiers->stream_info(), stream_id, "Stream", "stream_id");
        return *this;
    }

    insert_validator& require_kernel_symbol(
        writer_api::kernel_symbol_id_t kernel_symbol_id)
    {
        if(!m_identifiers->kernel_symbol_info().is_entry_registered(kernel_symbol_id))
        {
            throw std::runtime_error(fmt::format(
                "Kernel symbol not registered: kernel_id: {}", kernel_symbol_id));
        }
        return *this;
    }

    insert_validator& require_code_object(writer_api::code_object_id_t code_object_id)
    {
        if(!m_identifiers->code_object_info().is_entry_registered(code_object_id))
        {
            throw std::runtime_error(fmt::format(
                "Code object not registered: code_obj_id: {}", code_object_id));
        }
        return *this;
    }

    insert_validator& require_pmc(const writer_api::pmc_info_unique_id_t& pmc_unique_id)
    {
        if(!m_identifiers->pmc_info().is_entry_registered(pmc_unique_id))
        {
            throw std::runtime_error(
                fmt::format("PMC Info not registered: pmc_name: {}", pmc_unique_id.name));
        }
        return *this;
    }

    insert_validator& validate_optional_thread(
        const std::optional<writer_api::thread_id_t>& thread_id)
    {
        validate_optional(m_identifiers->thread_info(), thread_id, "Thread", "thread_id");
        return *this;
    }

    insert_validator& validate_optional_process(
        const std::optional<writer_api::process_id_t>& process_id)
    {
        validate_optional(m_identifiers->process_info(), process_id, "Process", "pid");
        return *this;
    }

    insert_validator& validate_optional_agent(
        const std::optional<writer_api::agent_unique_id_t>& agent_id,
        std::string_view                                    agent_role = "Agent")
    {
        validate_optional(m_identifiers->agent_info(), agent_id, agent_role, "agent_id");
        return *this;
    }

    insert_validator& validate_optional_queue(
        const std::optional<writer_api::queue_id_t>& queue_id)
    {
        validate_optional(m_identifiers->queue_info(), queue_id, "Queue", "queue_id");
        return *this;
    }

    insert_validator& validate_optional_stream(
        const std::optional<writer_api::stream_id_t>& stream_id)
    {
        validate_optional(m_identifiers->stream_info(), stream_id, "Stream", "stream_id");
        return *this;
    }

    [[nodiscard]] primary_key resolve_process_key(
        const std::optional<writer_api::process_id_t>& process_id) const
    {
        return m_identifiers->process_info().get_primary_key_value_for_entity(
            process_id.value());
    }

    [[nodiscard]] primary_key resolve_process_key(
        writer_api::process_id_t process_id) const
    {
        return m_identifiers->process_info().get_primary_key_value_for_entity(process_id);
    }

    [[nodiscard]] primary_key resolve_thread_key(
        const std::optional<writer_api::thread_id_t>& thread_id) const
    {
        return m_identifiers->thread_info().get_primary_key_value_for_entity(
            thread_id.value());
    }

    [[nodiscard]] primary_key resolve_agent_key(
        const std::optional<writer_api::agent_unique_id_t>& agent_id) const
    {
        return m_identifiers->agent_info().get_primary_key_value_for_entity(
            agent_id.value());
    }

    [[nodiscard]] primary_key resolve_agent_key(
        const writer_api::agent_unique_id_t& agent_id) const
    {
        return m_identifiers->agent_info().get_primary_key_value_for_entity(agent_id);
    }

    [[nodiscard]] primary_key resolve_queue_key(
        const std::optional<writer_api::queue_id_t>& queue_id) const
    {
        return m_identifiers->queue_info().get_primary_key_value_for_entity(
            queue_id.value());
    }

    [[nodiscard]] primary_key resolve_stream_key(
        const std::optional<writer_api::stream_id_t>& stream_id) const
    {
        return m_identifiers->stream_info().get_primary_key_value_for_entity(
            stream_id.value());
    }

    [[nodiscard]] primary_key resolve_pmc_key(
        const writer_api::pmc_info_unique_id_t& pmc_unique_id) const
    {
        return m_identifiers->pmc_info().get_primary_key_value_for_entity(pmc_unique_id);
    }

    [[nodiscard]] std::optional<primary_key> resolve_optional_process_key(
        const std::optional<writer_api::process_id_t>& process_id) const
    {
        return resolve_optional_key(m_identifiers->process_info(), process_id);
    }

    [[nodiscard]] std::optional<primary_key> resolve_optional_thread_key(
        const std::optional<writer_api::thread_id_t>& thread_id) const
    {
        return resolve_optional_key(m_identifiers->thread_info(), thread_id);
    }

    [[nodiscard]] std::optional<primary_key> resolve_optional_agent_key(
        const std::optional<writer_api::agent_unique_id_t>& agent_id) const
    {
        return resolve_optional_key(m_identifiers->agent_info(), agent_id);
    }

    [[nodiscard]] std::optional<primary_key> resolve_optional_queue_key(
        const std::optional<writer_api::queue_id_t>& queue_id) const
    {
        return resolve_optional_key(m_identifiers->queue_info(), queue_id);
    }

    [[nodiscard]] std::optional<primary_key> resolve_optional_stream_key(
        const std::optional<writer_api::stream_id_t>& stream_id) const
    {
        return resolve_optional_key(m_identifiers->stream_info(), stream_id);
    }

    [[nodiscard]] std::optional<primary_key> resolve_optional_string_key(
        const std::optional<std::string>& str) const
    {
        if(!str.has_value()) return std::nullopt;
        return m_identifiers->string_info().get_primary_key_value_for_entity(str.value());
    }

    [[nodiscard]] data_identifiers& identifiers() const { return *m_identifiers; }

private:
    template <typename Utility, typename EntityId>
    void validate_required(Utility&                       utility,
                           const std::optional<EntityId>& entity_id,
                           std::string_view               entity_name,
                           std::string_view               field_name)
    {
        if(!entity_id.has_value() || !utility.is_entry_registered(entity_id.value()))
        {
            throw std::runtime_error(fmt::format("{} not registered: {}: {}",
                                                 entity_name,
                                                 field_name,
                                                 to_error_string(entity_id)));
        }
    }

    template <typename Utility, typename EntityId>
    void validate_direct(Utility&         utility,
                         const EntityId&  entity_id,
                         std::string_view entity_name,
                         std::string_view field_name)
    {
        if(!utility.is_entry_registered(entity_id))
        {
            throw std::runtime_error(fmt::format("{} not registered: {}: {}",
                                                 entity_name,
                                                 field_name,
                                                 std::to_string(entity_id)));
        }
    }

    template <typename Utility, typename EntityId>
    void validate_optional(Utility&                       utility,
                           const std::optional<EntityId>& entity_id,
                           std::string_view               entity_name,
                           std::string_view               field_name)
    {
        if(entity_id.has_value() && !utility.is_entry_registered(entity_id.value()))
        {
            throw std::runtime_error(fmt::format("{} not registered: {}: {}",
                                                 entity_name,
                                                 field_name,
                                                 std::to_string(entity_id.value())));
        }
    }

    template <typename Utility, typename EntityId>
    [[nodiscard]] std::optional<size_t> resolve_optional_key(
        const Utility&                 utility,
        const std::optional<EntityId>& entity_id) const
    {
        if(!entity_id.has_value()) return std::nullopt;
        return utility.get_primary_key_value_for_entity(entity_id.value());
    }

    std::shared_ptr<data_identifiers> m_identifiers;
};
}  // namespace rocstorage
