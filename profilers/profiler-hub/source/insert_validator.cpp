// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "insert_validator.hpp"

#include "profiler-hub/writer_types.hpp"

#include "common/string_conversions.hpp"
#include "entity_registry.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace profiler_hub
{

namespace
{

template <typename Utility, typename EntityId>
void
validate_required(Utility&                       utility,
                  const std::optional<EntityId>& entity_id,
                  std::string_view               entity_name,
                  std::string_view               field_name)
{
    if(!entity_id.has_value() || !utility.is_entry_registered(entity_id.value()))
    {
        throw std::runtime_error(fmt::format(
            "{} not registered: {}: {}", entity_name, field_name, to_string(entity_id)));
    }
}

template <typename Utility, typename EntityId>
void
validate_direct(Utility&         utility,
                const EntityId&  entity_id,
                std::string_view entity_name,
                std::string_view field_name)
{
    if(!utility.is_entry_registered(entity_id))
    {
        throw std::runtime_error(fmt::format(
            "{} not registered: {}: {}", entity_name, field_name, to_string(entity_id)));
    }
}

template <typename Utility, typename EntityId>
void
validate_optional(Utility&                       utility,
                  const std::optional<EntityId>& entity_id,
                  std::string_view               entity_name,
                  std::string_view               field_name)
{
    if(entity_id.has_value() && !utility.is_entry_registered(entity_id.value()))
    {
        throw std::runtime_error(fmt::format("{} not registered: {}: {}",
                                             entity_name,
                                             field_name,
                                             to_string(entity_id.value())));
    }
}

template <typename Utility, typename EntityId>
[[nodiscard]] std::optional<size_t>
resolve_optional_key(const Utility& utility, const std::optional<EntityId>& entity_id)
{
    if(!entity_id.has_value())
    {
        return std::nullopt;
    }
    return { utility.get_primary_key_value_for_entity(entity_id.value()) };
}

template <typename EntityId>
const EntityId&
require_value(const std::optional<EntityId>& entity_id, std::string_view field_name)
{
    if(!entity_id.has_value())
    {
        throw std::invalid_argument(
            fmt::format("Required entity id is not set: {}", field_name));
    }
    return entity_id.value();
}

}  // namespace

insert_validator::insert_validator(std::shared_ptr<entity_registry> registry)
: m_registry(std::move(registry))
{}

insert_validator&
insert_validator::require_node(const std::optional<writer_types::node_id_t>& node_id)
{
    validate_required(m_registry->node_info(), node_id, "Node", "node_id");
    return *this;
}

insert_validator&
insert_validator::require_node(writer_types::node_id_t node_id)
{
    validate_direct(m_registry->node_info(), node_id, "Node", "node_id");
    return *this;
}

insert_validator&
insert_validator::require_process(
    const std::optional<writer_types::process_id_t>& process_id)
{
    validate_required(m_registry->process_info(), process_id, "Process", "pid");
    return *this;
}

insert_validator&
insert_validator::require_process(writer_types::process_id_t process_id)
{
    validate_direct(m_registry->process_info(), process_id, "Process", "pid");
    return *this;
}

insert_validator&
insert_validator::require_thread(
    const std::optional<writer_types::thread_id_t>& thread_id)
{
    validate_required(m_registry->thread_info(), thread_id, "Thread", "thread_id");
    return *this;
}

insert_validator&
insert_validator::require_agent(
    const std::optional<writer_types::agent_unique_id_t>& agent_id)
{
    validate_required(m_registry->agent_info(), agent_id, "Agent", "agent_id");
    return *this;
}

insert_validator&
insert_validator::require_agent(const writer_types::agent_unique_id_t& agent_id)
{
    validate_direct(m_registry->agent_info(), agent_id, "Agent", "agent_id");
    return *this;
}

insert_validator&
insert_validator::require_queue(const std::optional<writer_types::queue_id_t>& queue_id)
{
    validate_required(m_registry->queue_info(), queue_id, "Queue", "queue_id");
    return *this;
}

insert_validator&
insert_validator::require_stream(
    const std::optional<writer_types::stream_id_t>& stream_id)
{
    validate_required(m_registry->stream_info(), stream_id, "Stream", "stream_id");
    return *this;
}

insert_validator&
insert_validator::require_kernel_symbol(writer_types::kernel_symbol_id_t kernel_symbol_id)
{
    validate_direct(
        m_registry->kernel_symbol_info(), kernel_symbol_id, "Kernel symbol", "kernel_id");
    return *this;
}

insert_validator&
insert_validator::require_code_object(writer_types::code_object_id_t code_object_id)
{
    validate_direct(
        m_registry->code_object_info(), code_object_id, "Code object", "code_obj_id");
    return *this;
}

insert_validator&
insert_validator::require_pmc(const writer_types::pmc_info_unique_id_t& pmc_unique_id)
{
    validate_direct(m_registry->pmc_info(), pmc_unique_id, "PMC Info", "pmc_id");
    return *this;
}

insert_validator&
insert_validator::validate_optional_thread(
    const std::optional<writer_types::thread_id_t>& thread_id)
{
    validate_optional(m_registry->thread_info(), thread_id, "Thread", "thread_id");
    return *this;
}

insert_validator&
insert_validator::validate_optional_process(
    const std::optional<writer_types::process_id_t>& process_id)
{
    validate_optional(m_registry->process_info(), process_id, "Process", "pid");
    return *this;
}

insert_validator&
insert_validator::validate_optional_agent(
    const std::optional<writer_types::agent_unique_id_t>& agent_id,
    std::string_view                                      agent_role)
{
    validate_optional(m_registry->agent_info(), agent_id, agent_role, "agent_id");
    return *this;
}

insert_validator&
insert_validator::validate_optional_queue(
    const std::optional<writer_types::queue_id_t>& queue_id)
{
    validate_optional(m_registry->queue_info(), queue_id, "Queue", "queue_id");
    return *this;
}

insert_validator&
insert_validator::validate_optional_stream(
    const std::optional<writer_types::stream_id_t>& stream_id)
{
    validate_optional(m_registry->stream_info(), stream_id, "Stream", "stream_id");
    return *this;
}

primary_key_t
insert_validator::resolve_process_key(
    const std::optional<writer_types::process_id_t>& process_id) const
{
    return m_registry->process_info().get_primary_key_value_for_entity(
        require_value(process_id, "process_id"));
}

primary_key_t
insert_validator::resolve_process_key(writer_types::process_id_t process_id) const
{
    return m_registry->process_info().get_primary_key_value_for_entity(process_id);
}

primary_key_t
insert_validator::resolve_thread_key(
    const std::optional<writer_types::thread_id_t>& thread_id) const
{
    return m_registry->thread_info().get_primary_key_value_for_entity(
        require_value(thread_id, "thread_id"));
}

primary_key_t
insert_validator::resolve_agent_key(
    const std::optional<writer_types::agent_unique_id_t>& agent_id) const
{
    return m_registry->agent_info().get_primary_key_value_for_entity(
        require_value(agent_id, "agent_id"));
}

primary_key_t
insert_validator::resolve_agent_key(const writer_types::agent_unique_id_t& agent_id) const
{
    return m_registry->agent_info().get_primary_key_value_for_entity(agent_id);
}

primary_key_t
insert_validator::resolve_queue_key(
    const std::optional<writer_types::queue_id_t>& queue_id) const
{
    return m_registry->queue_info().get_primary_key_value_for_entity(
        require_value(queue_id, "queue_id"));
}

primary_key_t
insert_validator::resolve_stream_key(
    const std::optional<writer_types::stream_id_t>& stream_id) const
{
    return m_registry->stream_info().get_primary_key_value_for_entity(
        require_value(stream_id, "stream_id"));
}

primary_key_t
insert_validator::resolve_pmc_key(
    const writer_types::pmc_info_unique_id_t& pmc_unique_id) const
{
    return m_registry->pmc_info().get_primary_key_value_for_entity(pmc_unique_id);
}

std::optional<primary_key_t>
insert_validator::resolve_optional_process_key(
    const std::optional<writer_types::process_id_t>& process_id) const
{
    return resolve_optional_key(m_registry->process_info(), process_id);
}

std::optional<primary_key_t>
insert_validator::resolve_optional_thread_key(
    const std::optional<writer_types::thread_id_t>& thread_id) const
{
    return resolve_optional_key(m_registry->thread_info(), thread_id);
}

std::optional<primary_key_t>
insert_validator::resolve_optional_agent_key(
    const std::optional<writer_types::agent_unique_id_t>& agent_id) const
{
    return resolve_optional_key(m_registry->agent_info(), agent_id);
}

std::optional<primary_key_t>
insert_validator::resolve_optional_queue_key(
    const std::optional<writer_types::queue_id_t>& queue_id) const
{
    return resolve_optional_key(m_registry->queue_info(), queue_id);
}

std::optional<primary_key_t>
insert_validator::resolve_optional_stream_key(
    const std::optional<writer_types::stream_id_t>& stream_id) const
{
    return resolve_optional_key(m_registry->stream_info(), stream_id);
}

std::optional<primary_key_t>
insert_validator::resolve_optional_string_key(const std::optional<std::string>& str) const
{
    if(!str.has_value()) return std::nullopt;
    return m_registry->string_info().get_primary_key_value_for_entity(str.value());
}

entity_registry&
insert_validator::registry() const
{
    return *m_registry;
}

}  // namespace profiler_hub
