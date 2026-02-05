// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "writer_impl.hpp"
#include "data_storage/insert_statements.hpp"
#include "entity_registry.hpp"
#include "insert_validator.hpp"
#include "primary_key_providers.hpp"
#include "rocstorage/storage.hpp"
#include "rocstorage/writer.hpp"
#include "rocstorage/writer_types.hpp"
#include "spdlog/fmt/bundled/core.h"
#include "storage_impl.hpp"

#include "common/string_conversions.hpp"
#include "debug.hpp"
#include "json_serializers.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace rocstorage
{

namespace
{

/**
 * @brief Check if entity is already registered and log warning if so
 * @return true if already registered (caller should return early), false otherwise
 */
template <typename Utility, typename Entity>
[[nodiscard]] bool
is_already_registered(Utility& utility, const Entity& entity)
{
    if(utility.is_entry_registered(get_key(entity)))
    {
        LOG_WARNING("{} already registered", rocstorage::to_string(entity));
        return true;
    }
    return false;
}

}  // namespace

writer_t::impl::impl(std::unique_ptr<rocstorage::storage_t> storage)
: m_storage(std::move(storage))
, m_database(m_storage->m_impl->create_database(storage_t::impl::storage_type_t::write))
, m_uuid(m_storage->m_impl->get_uuid())
, m_entity_registry(std::make_shared<entity_registry>())
, m_key_providers(std::make_shared<primary_key_providers>())
, m_validator(std::make_shared<insert_validator>(m_entity_registry))
{
    if(!m_database)
    {
        throw std::invalid_argument("Provided pointer to a non-existing database!");
    }

    if(m_uuid.empty())
    {
        throw std::invalid_argument("Empty UUID provided!");
    }

    m_database->initialize_schema();
    m_insert_statements =
        std::make_unique<data_storage::schema_v3::insert_statements>(m_database, m_uuid);
}

void
writer_t::impl::register_node_info(const writer_types::node_info_t& node_info)
{
    auto& node_info_utility = m_entity_registry->node_info();
    if(is_already_registered(node_info_utility, node_info)) return;

    m_insert_statements->node_info_statement()(node_info.node_id,
                                               node_info.hash,
                                               node_info.machine_id,
                                               node_info.system_name,
                                               node_info.hostname,
                                               node_info.release,
                                               node_info.version,
                                               node_info.hardware_name,
                                               node_info.domain_name);

    node_info_utility.emplace_entity(node_info.node_id);

    LOG_TRACE("Registered node info: {}", rocstorage::to_string(node_info));
}

void
writer_t::impl::register_process_info(const writer_types::process_info_t& process_info)
{
    auto& process_info_utility = m_entity_registry->process_info();
    if(is_already_registered(process_info_utility, process_info)) return;

    m_validator->require_node(process_info.node_id);

    const auto primary_key = m_key_providers->process_info().get_primary_key_value();

    m_insert_statements->process_info_statement()(primary_key,
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

    LOG_TRACE("Registered process info: {}", rocstorage::to_string(process_info));
}

void
writer_t::impl::register_agent_info(const writer_types::agent_info_t& agent_info)
{
    auto& agent_info_utility = m_entity_registry->agent_info();
    if(is_already_registered(agent_info_utility, agent_info)) return;

    m_validator->require_node(agent_info.node_id).require_process(agent_info.process_id);

    const std::string_view agent_type{ agent_info.unique_id.agent_type };
    if(agent_type != "CPU" && agent_type != "GPU")
    {
        throw std::invalid_argument(fmt::format(
            "Invalid agent type: {}. Type can be NULL, CPU, or GPU.", agent_type));
    }

    const auto process_pk  = m_validator->resolve_process_key(agent_info.process_id);
    const auto primary_key = m_key_providers->agent_info().get_primary_key_value();

    m_insert_statements->agent_info_statement()(primary_key,
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

    LOG_TRACE("Registered agent info: {}", rocstorage::to_string(agent_info));
}

void
writer_t::impl::register_pmc_info(const writer_types::pmc_info_t& pmc_info)
{
    auto& pmc_info_utility = m_entity_registry->pmc_info();
    if(is_already_registered(pmc_info_utility, pmc_info)) return;

    m_validator->require_node(pmc_info.node_id)
        .require_process(pmc_info.process_id)
        .require_agent(*pmc_info.unique_id.agent_id);

    const auto process_pk  = m_validator->resolve_process_key(pmc_info.process_id);
    const auto agent_pk    = m_validator->resolve_agent_key(*pmc_info.unique_id.agent_id);
    const auto primary_key = m_key_providers->pmc_info().get_primary_key_value();

    m_insert_statements->pmc_info_statement()(primary_key,
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

    LOG_TRACE("Registered pmc info: {}", rocstorage::to_string(pmc_info));
}

void
writer_t::impl::register_thread_info(const writer_types::thread_info_t& thread_info)
{
    auto& thread_info_utility = m_entity_registry->thread_info();
    if(is_already_registered(thread_info_utility, thread_info)) return;

    m_validator->require_node(thread_info.node_id)
        .require_process(thread_info.process_id);

    const auto process_pk  = m_validator->resolve_process_key(thread_info.process_id);
    const auto primary_key = m_key_providers->thread_info().get_primary_key_value();

    m_insert_statements->thread_info_statement()(primary_key,
                                                 thread_info.node_id,
                                                 thread_info.parent_process_id,
                                                 process_pk,
                                                 thread_info.thread_id,
                                                 thread_info.name,
                                                 thread_info.start,
                                                 thread_info.end,
                                                 thread_info.extdata);

    thread_info_utility.emplace_entity(thread_info.thread_id, primary_key);

    LOG_TRACE("Registered thread info: {}", rocstorage::to_string(thread_info));
}

void
writer_t::impl::register_stream_info(const writer_types::stream_info_t& stream_info)
{
    // assert("TODO Fix stream id");
    auto& stream_info_utility = m_entity_registry->stream_info();
    if(is_already_registered(stream_info_utility, stream_info)) return;

    m_validator->require_node(stream_info.node_id)
        .require_process(stream_info.process_id);

    const auto process_pk  = m_validator->resolve_process_key(stream_info.process_id);
    const auto primary_key = m_key_providers->stream_info().get_primary_key_value();

    m_insert_statements->stream_info_statement()(primary_key,
                                                 stream_info.node_id,
                                                 process_pk,
                                                 stream_info.name,
                                                 stream_info.extdata);

    stream_info_utility.emplace_entity(stream_info.stream_id, primary_key);

    LOG_TRACE("Registered stream info: {}", rocstorage::to_string(stream_info));
}

void
writer_t::impl::register_queue_info(const writer_types::queue_info_t& queue_info)
{
    // assert("TODO Fix queue id");
    auto& queue_info_utility = m_entity_registry->queue_info();
    if(is_already_registered(queue_info_utility, queue_info)) return;

    m_validator->require_node(queue_info.node_id).require_process(queue_info.process_id);

    const auto process_pk  = m_validator->resolve_process_key(queue_info.process_id);
    const auto primary_key = m_key_providers->queue_info().get_primary_key_value();

    m_insert_statements->queue_info_statement()(
        primary_key, queue_info.node_id, process_pk, queue_info.name, queue_info.extdata);

    queue_info_utility.emplace_entity(queue_info.queue_id, primary_key);

    LOG_TRACE("Registered queue info: {}", rocstorage::to_string(queue_info));
}

void
writer_t::impl::register_code_object_info(
    const writer_types::code_object_info_t& code_object_info)
{
    auto& code_object_info_utility = m_entity_registry->code_object_info();
    if(is_already_registered(code_object_info_utility, code_object_info)) return;

    m_validator->require_node(code_object_info.node_id)
        .require_process(code_object_info.process_id)
        .validate_optional_agent(code_object_info.agent_id);

    const auto process_pk = m_validator->resolve_process_key(code_object_info.process_id);
    const auto agent_pk =
        m_validator->resolve_optional_agent_key(code_object_info.agent_id);

    m_insert_statements->code_object_info_statement()(code_object_info.id,
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

    LOG_TRACE("Registered code object info: {}", rocstorage::to_string(code_object_info));
}

void
writer_t::impl::register_kernel_symbol_info(
    const writer_types::kernel_symbol_info_t& kernel_symbol_info)
{
    auto& kernel_symbol_info_utility = m_entity_registry->kernel_symbol_info();
    if(is_already_registered(kernel_symbol_info_utility, kernel_symbol_info)) return;

    m_validator->require_node(kernel_symbol_info.node_id)
        .require_process(kernel_symbol_info.process_id)
        .require_code_object(kernel_symbol_info.code_obj_id);

    const auto process_pk =
        m_validator->resolve_process_key(kernel_symbol_info.process_id);

    m_insert_statements->kernel_symbol_info_statement()(
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
              rocstorage::to_string(kernel_symbol_info));
}

void
writer_t::impl::register_track_info(const writer_types::track_info_t& track)
{
    auto& track_info_utility = m_entity_registry->track_info();
    if(is_already_registered(track_info_utility, track)) return;

    m_validator->require_node(track.node_id)
        .validate_optional_process(track.process_id)
        .validate_optional_thread(track.thread_id);

    if(track.name.has_value() &&
       !m_entity_registry->string_info().is_entry_registered(track.name.value()))
    {
        register_string(track.name.value());
    }

    const auto process_pk  = m_validator->resolve_optional_process_key(track.process_id);
    const auto thread_pk   = m_validator->resolve_optional_thread_key(track.thread_id);
    const auto string_pk   = m_validator->resolve_optional_string_key(track.name);
    const auto primary_key = m_key_providers->track_info().get_primary_key_value();

    m_insert_statements->track_info_statement()(
        primary_key, track.node_id, process_pk, thread_pk, string_pk, track.extdata);

    track_info_utility.emplace_entity(track, primary_key);

    LOG_TRACE("Registered track info: {}", rocstorage::to_string(track));
}

void
writer_t::impl::register_string(const char* str)
{
    auto& string_info_utility = m_entity_registry->string_info();

    if(str == nullptr)
    {
        throw std::runtime_error("Trying to register string that is null");
    }

    const auto is_string_registered = string_info_utility.is_entry_registered(str);

    if(is_string_registered)
    {
        LOG_WARNING("String already registered: str: {}", str);
        return;
    }

    const auto primary_key = m_key_providers->string_info().get_primary_key_value();

    m_insert_statements->string_statement()(primary_key, str);

    string_info_utility.emplace_entity(str, primary_key);

    LOG_TRACE("Registered string: {}", str);
}

// --------------------- Data Tables ---------------------

primary_key_t
writer_t::impl::insert_event(const writer_types::event_data_t& event_data)
{
    auto& string_info_utility = m_entity_registry->string_info();

    const auto is_string_registered =
        string_info_utility.is_entry_registered(event_data.event_category);

    if(!is_string_registered)
    {
        register_string(event_data.event_category);
    }

    const auto event_category_primary_key =
        string_info_utility.get_primary_key_value_for_entity(event_data.event_category);
    const auto primary_key = m_key_providers->event_data().get_primary_key_value();

    m_insert_statements->event_statement()(
        primary_key,
        event_category_primary_key,
        event_data.stack_id,
        event_data.parent_stack_id,
        event_data.correlation_id,
        json_serializers::serialize_call_stack(event_data.call_stack).c_str(),
        json_serializers::serialize_source_context(event_data.line_info_list).c_str(),
        event_data.extdata);

    return primary_key;
}

void
writer_t::impl::insert_sample(const writer_types::sample_data_t& sample_data,
                              const primary_key_t&               event_primary_key)
{
    auto& track_info_utility = m_entity_registry->track_info();

    if(!track_info_utility.is_entry_registered(sample_data.track))
    {
        const auto* const track_name_print_value = sample_data.track.name.has_value()
                                                       ? sample_data.track.name.value()
                                                       : "[NULL]";

        throw std::runtime_error(
            fmt::format("Track not registered for Sample Data: track_name: {}",
                        track_name_print_value));
    }

    const auto track_primary_key =
        track_info_utility.get_primary_key_value_for_entity(sample_data.track);

    const auto primary_key = m_key_providers->sample_data().get_primary_key_value();
    m_insert_statements->sample_statement()(primary_key,
                                            track_primary_key,
                                            sample_data.timestamp,
                                            event_primary_key,
                                            sample_data.extdata);
}

void
writer_t::impl::insert_arg(const writer_types::arg_data_t& arg_data,
                           primary_key_t                   event_id)
{
    auto& string_info_utility = m_entity_registry->string_info();

    if(arg_data.type == nullptr || arg_data.name == nullptr)
    {
        throw std::runtime_error(
            fmt::format("Type or name is null for Arg Data: type: {}, name: {}",
                        arg_data.type,
                        arg_data.name));
    }

    const auto is_string_registered =
        string_info_utility.is_entry_registered(arg_data.type);

    if(!is_string_registered)
    {
        register_string(arg_data.type);
    }

    const auto primary_key = m_key_providers->arg().get_primary_key_value();

    m_insert_statements->arg_statement()(primary_key,
                                         event_id,
                                         arg_data.position,
                                         arg_data.type,
                                         arg_data.name,
                                         arg_data.value,
                                         arg_data.extdata);
}

void
writer_t::impl::insert_region_data(
    const writer_types::region_data_t&       region_data,
    const writer_types::trace_environment_t& trace_environment)
{
    auto transaction_block = m_database->create_transaction_block();

    m_validator->require_node(trace_environment.node_id)
        .require_process(trace_environment.process_id)
        .require_thread(trace_environment.thread_id);

    if(!region_data.event.has_value() && !region_data.args.empty())
    {
        throw std::runtime_error(fmt::format(
            "Writing args require providing event data for correlation: name: {}",
            region_data.name));
    }

    auto& string_info_utility = m_entity_registry->string_info();
    if(!string_info_utility.is_entry_registered(region_data.name))
    {
        register_string(region_data.name);
    }

    const auto process_pk =
        m_validator->resolve_process_key(trace_environment.process_id);
    const auto thread_pk = m_validator->resolve_thread_key(trace_environment.thread_id);
    const auto name_pk =
        string_info_utility.get_primary_key_value_for_entity(region_data.name);

    std::optional<primary_key_t> event_pk = std::nullopt;
    if(region_data.event.has_value())
    {
        event_pk = insert_event(region_data.event.value());
    }

    const auto primary_key = m_key_providers->region_data().get_primary_key_value();

    m_insert_statements->region_statement()(primary_key,
                                            trace_environment.node_id.value(),
                                            process_pk,
                                            thread_pk,
                                            region_data.start_timestamp,
                                            region_data.end_timestamp,
                                            name_pk,
                                            event_pk,
                                            region_data.extdata);

    if(event_pk.has_value())
    {
        for(const auto& arg : region_data.args)
        {
            insert_arg(arg, event_pk.value());
        }
    }

    if(trace_environment.track_name.has_value() && event_pk.has_value())
    {
        const writer_types::track_info_t track_info = {
            trace_environment.track_name.value(),
            nullptr,
            trace_environment.node_id.value(),
            trace_environment.process_id.value(),
            trace_environment.thread_id.value()
        };
        const writer_types::sample_data_t sample_data = { region_data.start_timestamp,
                                                          track_info,
                                                          "{}" };
        insert_sample(sample_data, event_pk.value());
    }
}

void
writer_t::impl::insert_pmc_event_data(
    const writer_types::pmc_event_data_t&     pmc_event_data,
    const writer_types::pmc_info_unique_id_t& pmc_unique_id)
{
    auto transaction_block = m_database->create_transaction_block();

    m_validator->require_pmc(pmc_unique_id);

    const auto pmc_pk = m_validator->resolve_pmc_key(pmc_unique_id);

    std::optional<primary_key_t> event_pk = std::nullopt;
    if(pmc_event_data.event.has_value())
    {
        event_pk = insert_event(pmc_event_data.event.value());
    }

    const auto primary_key = m_key_providers->pmc_event_data().get_primary_key_value();

    m_insert_statements->pmc_event_statement()(
        primary_key, event_pk, pmc_pk, pmc_event_data.value, pmc_event_data.extdata);
}

void
writer_t::impl::insert_kernel_dispatch_data(
    const writer_types::kernel_dispatch_data_t& kernel_dispatch_data,
    const writer_types::trace_environment_t&    trace_environment)
{
    auto transaction_block = m_database->create_transaction_block();

    m_validator->require_node(trace_environment.node_id)
        .require_process(trace_environment.process_id)
        .require_thread(trace_environment.thread_id)
        .require_agent(trace_environment.agent_id)
        .require_queue(trace_environment.queue_id)
        .require_stream(trace_environment.stream_id)
        .require_kernel_symbol(kernel_dispatch_data.kernel_symbol_id);

    auto& string_info_utility = m_entity_registry->string_info();
    if(!string_info_utility.is_entry_registered(kernel_dispatch_data.name))
    {
        register_string(kernel_dispatch_data.name);
    }

    const auto process_pk =
        m_validator->resolve_process_key(trace_environment.process_id);
    const auto thread_pk =
        m_validator->resolve_optional_thread_key(trace_environment.thread_id);
    const auto agent_pk  = m_validator->resolve_agent_key(trace_environment.agent_id);
    const auto queue_pk  = m_validator->resolve_queue_key(trace_environment.queue_id);
    const auto stream_pk = m_validator->resolve_stream_key(trace_environment.stream_id);
    const auto name_pk =
        string_info_utility.get_primary_key_value_for_entity(kernel_dispatch_data.name);

    std::optional<primary_key_t> event_pk = std::nullopt;
    if(kernel_dispatch_data.event.has_value())
    {
        event_pk = insert_event(kernel_dispatch_data.event.value());
    }

    const auto primary_key =
        m_key_providers->kernel_dispatch_data().get_primary_key_value();

    m_insert_statements->kernel_dispatch_statement()(
        primary_key,
        trace_environment.node_id.value(),
        process_pk,
        thread_pk,
        agent_pk,
        kernel_dispatch_data.kernel_symbol_id,
        kernel_dispatch_data.dispatch_id,
        queue_pk,
        stream_pk,
        kernel_dispatch_data.start_timestamp,
        kernel_dispatch_data.end_timestamp,
        kernel_dispatch_data.private_segment_size,
        kernel_dispatch_data.group_segment_size,
        kernel_dispatch_data.workgroup_size_x,
        kernel_dispatch_data.workgroup_size_y,
        kernel_dispatch_data.workgroup_size_z,
        kernel_dispatch_data.grid_size_x,
        kernel_dispatch_data.grid_size_y,
        kernel_dispatch_data.grid_size_z,
        name_pk,
        event_pk,
        kernel_dispatch_data.extdata);

    if(trace_environment.track_name.has_value() && event_pk.has_value())
    {
        const writer_types::track_info_t track_info = {
            trace_environment.track_name.value(),
            nullptr,
            trace_environment.node_id.value(),
            trace_environment.process_id.value(),
            trace_environment.thread_id.value()
        };
        const writer_types::sample_data_t sample_data = {
            kernel_dispatch_data.start_timestamp, track_info, "{}"
        };
        insert_sample(sample_data, event_pk.value());
    }
}

void
writer_t::impl::insert_memory_copy_data(
    const writer_types::memory_copy_data_t&  memory_copy_data,
    const writer_types::trace_environment_t& trace_environment)
{
    auto transaction_block = m_database->create_transaction_block();

    m_validator->require_node(trace_environment.node_id)
        .require_process(trace_environment.process_id)
        .validate_optional_thread(trace_environment.thread_id)
        .validate_optional_agent(memory_copy_data.src_agent_id, "Source agent")
        .validate_optional_agent(memory_copy_data.dst_agent_id, "Destination agent")
        .validate_optional_queue(trace_environment.queue_id)
        .validate_optional_stream(trace_environment.stream_id);

    auto& string_info_utility = m_entity_registry->string_info();
    if(!string_info_utility.is_entry_registered(memory_copy_data.name))
    {
        register_string(memory_copy_data.name);
    }
    if(memory_copy_data.region_name != nullptr &&
       !string_info_utility.is_entry_registered(memory_copy_data.region_name))
    {
        register_string(memory_copy_data.region_name);
    }

    const auto process_pk =
        m_validator->resolve_process_key(trace_environment.process_id);
    const auto thread_pk =
        m_validator->resolve_optional_thread_key(trace_environment.thread_id);
    const auto src_agent_pk =
        m_validator->resolve_optional_agent_key(memory_copy_data.src_agent_id);
    const auto dst_agent_pk =
        m_validator->resolve_optional_agent_key(memory_copy_data.dst_agent_id);
    const auto queue_pk =
        m_validator->resolve_optional_queue_key(trace_environment.queue_id);
    const auto stream_pk =
        m_validator->resolve_optional_stream_key(trace_environment.stream_id);
    const auto name_pk =
        string_info_utility.get_primary_key_value_for_entity(memory_copy_data.name);

    std::optional<primary_key_t> event_pk = std::nullopt;
    if(memory_copy_data.event.has_value())
    {
        event_pk = insert_event(memory_copy_data.event.value());
    }

    std::optional<primary_key_t> region_name_pk = std::nullopt;
    if(memory_copy_data.region_name != nullptr)
    {
        region_name_pk = string_info_utility.get_primary_key_value_for_entity(
            memory_copy_data.region_name);
    }

    const auto primary_key = m_key_providers->memory_copy_data().get_primary_key_value();

    m_insert_statements->memory_copy_statement()(primary_key,
                                                 trace_environment.node_id.value(),
                                                 process_pk,
                                                 thread_pk,
                                                 memory_copy_data.start_timestamp,
                                                 memory_copy_data.end_timestamp,
                                                 name_pk,
                                                 dst_agent_pk,
                                                 memory_copy_data.dst_address,
                                                 src_agent_pk,
                                                 memory_copy_data.src_address,
                                                 memory_copy_data.size,
                                                 queue_pk,
                                                 stream_pk,
                                                 region_name_pk,
                                                 event_pk,
                                                 memory_copy_data.extdata);

    if(trace_environment.track_name.has_value() && event_pk.has_value())
    {
        const writer_types::track_info_t track_info = {
            trace_environment.track_name.value(),
            nullptr,
            trace_environment.node_id.value(),
            trace_environment.process_id.value(),
            trace_environment.thread_id.value()
        };
        const writer_types::sample_data_t sample_data = {
            memory_copy_data.start_timestamp, track_info, "{}"
        };
        insert_sample(sample_data, event_pk.value());
    }
}

void
writer_t::impl::insert_memory_alloc_data(
    const writer_types::memory_alloc_data_t& memory_alloc_data,
    const writer_types::trace_environment_t& trace_environment)
{
    auto transaction_block = m_database->create_transaction_block();

    m_validator->require_node(trace_environment.node_id)
        .require_process(trace_environment.process_id)
        .validate_optional_thread(trace_environment.thread_id)
        .validate_optional_agent(trace_environment.agent_id)
        .validate_optional_queue(trace_environment.queue_id)
        .validate_optional_stream(trace_environment.stream_id);

    if(memory_alloc_data.type != nullptr)
    {
        constexpr std::array<std::string_view, 4> allowed_types = {
            "ALLOC", "FREE", "REALLOC", "RECLAIM"
        };
        if(std::find(allowed_types.begin(),
                     allowed_types.end(),
                     memory_alloc_data.type) == allowed_types.end())
        {
            throw std::runtime_error(fmt::format(
                "Invalid type value for Memory Alloc Data: type: {}. Allowed: {}",
                memory_alloc_data.type,
                allowed_types));
        }
    }

    if(memory_alloc_data.level != nullptr)
    {
        constexpr std::array<std::string_view, 3> allowed_levels = { "REAL",
                                                                     "VIRTUAL",
                                                                     "SCRATCH" };
        if(std::find(allowed_levels.begin(),
                     allowed_levels.end(),
                     memory_alloc_data.level) == allowed_levels.end())
        {
            throw std::runtime_error(fmt::format(
                "Invalid level value for Memory Alloc Data: level: {}. Allowed: {}",
                memory_alloc_data.level,
                allowed_levels));
        }
    }

    const auto process_pk =
        m_validator->resolve_process_key(trace_environment.process_id);
    const auto thread_pk =
        m_validator->resolve_optional_thread_key(trace_environment.thread_id);
    const auto agent_pk =
        m_validator->resolve_optional_agent_key(trace_environment.agent_id);
    const auto queue_pk =
        m_validator->resolve_optional_queue_key(trace_environment.queue_id);
    const auto stream_pk =
        m_validator->resolve_optional_stream_key(trace_environment.stream_id);

    std::optional<primary_key_t> event_pk = std::nullopt;
    if(memory_alloc_data.event.has_value())
    {
        event_pk = insert_event(memory_alloc_data.event.value());
    }

    const auto primary_key = m_key_providers->memory_alloc_data().get_primary_key_value();

    m_insert_statements->memory_alloc_statement()(primary_key,
                                                  trace_environment.node_id.value(),
                                                  process_pk,
                                                  thread_pk,
                                                  agent_pk,
                                                  memory_alloc_data.type,
                                                  memory_alloc_data.level,
                                                  memory_alloc_data.start_timestamp,
                                                  memory_alloc_data.end_timestamp,
                                                  memory_alloc_data.address,
                                                  memory_alloc_data.size,
                                                  queue_pk,
                                                  stream_pk,
                                                  event_pk,
                                                  memory_alloc_data.extdata);

    if(trace_environment.track_name.has_value() && event_pk.has_value())
    {
        const writer_types::track_info_t track_info = {
            trace_environment.track_name.value(),
            nullptr,
            trace_environment.node_id.value(),
            trace_environment.process_id.value(),
            trace_environment.thread_id.value()
        };
        const writer_types::sample_data_t sample_data = {
            memory_alloc_data.start_timestamp, track_info, "{}"
        };
        insert_sample(sample_data, event_pk.value());
    }
}

void
writer_t::impl::flush_in_memory_data_to_disk()
{
    m_database->flush();
}

}  // namespace rocstorage
