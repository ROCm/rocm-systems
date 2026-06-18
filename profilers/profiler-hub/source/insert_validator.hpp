// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/writer_types.hpp"

#include "entity_registry.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace profiler_hub
{

class insert_validator
{
public:
    explicit insert_validator(std::shared_ptr<entity_registry> registry);

    insert_validator& require_node(const std::optional<writer_types::node_id_t>& node_id);
    insert_validator& require_node(writer_types::node_id_t node_id);

    insert_validator& require_process(
        const std::optional<writer_types::process_id_t>& process_id);
    insert_validator& require_process(writer_types::process_id_t process_id);

    insert_validator& require_thread(
        const std::optional<writer_types::thread_id_t>& thread_id);

    insert_validator& require_agent(
        const std::optional<writer_types::agent_unique_id_t>& agent_id);
    insert_validator& require_agent(const writer_types::agent_unique_id_t& agent_id);

    insert_validator& require_queue(
        const std::optional<writer_types::queue_id_t>& queue_id);

    insert_validator& require_stream(
        const std::optional<writer_types::stream_id_t>& stream_id);

    insert_validator& require_kernel_symbol(
        writer_types::kernel_symbol_id_t kernel_symbol_id);

    insert_validator& require_code_object(writer_types::code_object_id_t code_object_id);

    insert_validator& require_pmc(
        const writer_types::pmc_info_unique_id_t& pmc_unique_id);

    insert_validator& validate_optional_thread(
        const std::optional<writer_types::thread_id_t>& thread_id);

    insert_validator& validate_optional_process(
        const std::optional<writer_types::process_id_t>& process_id);

    insert_validator& validate_optional_agent(
        const std::optional<writer_types::agent_unique_id_t>& agent_id,
        std::string_view                                      agent_role = "Agent");

    insert_validator& validate_optional_queue(
        const std::optional<writer_types::queue_id_t>& queue_id);

    insert_validator& validate_optional_stream(
        const std::optional<writer_types::stream_id_t>& stream_id);

    [[nodiscard]] primary_key_t resolve_process_key(
        const std::optional<writer_types::process_id_t>& process_id) const;

    [[nodiscard]] primary_key_t resolve_process_key(
        writer_types::process_id_t process_id) const;

    [[nodiscard]] primary_key_t resolve_thread_key(
        const std::optional<writer_types::thread_id_t>& thread_id) const;

    [[nodiscard]] primary_key_t resolve_agent_key(
        const std::optional<writer_types::agent_unique_id_t>& agent_id) const;

    [[nodiscard]] primary_key_t resolve_agent_key(
        const writer_types::agent_unique_id_t& agent_id) const;

    [[nodiscard]] primary_key_t resolve_queue_key(
        const std::optional<writer_types::queue_id_t>& queue_id) const;

    [[nodiscard]] primary_key_t resolve_stream_key(
        const std::optional<writer_types::stream_id_t>& stream_id) const;

    [[nodiscard]] primary_key_t resolve_pmc_key(
        const writer_types::pmc_info_unique_id_t& pmc_unique_id) const;

    [[nodiscard]] std::optional<primary_key_t> resolve_optional_process_key(
        const std::optional<writer_types::process_id_t>& process_id) const;

    [[nodiscard]] std::optional<primary_key_t> resolve_optional_thread_key(
        const std::optional<writer_types::thread_id_t>& thread_id) const;

    [[nodiscard]] std::optional<primary_key_t> resolve_optional_agent_key(
        const std::optional<writer_types::agent_unique_id_t>& agent_id) const;

    [[nodiscard]] std::optional<primary_key_t> resolve_optional_queue_key(
        const std::optional<writer_types::queue_id_t>& queue_id) const;

    [[nodiscard]] std::optional<primary_key_t> resolve_optional_stream_key(
        const std::optional<writer_types::stream_id_t>& stream_id) const;

    [[nodiscard]] std::optional<primary_key_t> resolve_optional_string_key(
        const std::optional<std::string>& str) const;

    [[nodiscard]] entity_registry& registry() const;

private:
    std::shared_ptr<entity_registry> m_registry;
};

}  // namespace profiler_hub
