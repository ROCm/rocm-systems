// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>
#include <optional>
#include <string_view>

namespace rocprofsys::policies::profiler_hub
{

/**
 * Concepts for the data records passed to a profiler-hub writer.
 *
 * Each concept takes a `Types` type (the writer) exposing the record types as nested
 * aliases (e.g. `Types::node_info_t`) and checks the fields `rocpd_processor_t` assigns.
 * String-like fields are non-owning views; nullable fields are `std::optional`.
 */

template <typename Id>
concept writer_id_policy = std::unsigned_integral<Id>;

template <typename Types>
concept agent_unique_id_policy =
    requires(typename Types::agent_unique_id_t& id,
             std::optional<std::string_view> agent_type, std::size_t type_index) {
        id.agent_type = agent_type;
        id.type_index = type_index;
        { id == id } -> std::convertible_to<bool>;
    };

template <typename Types>
concept trace_environment_policy =
    requires(typename Types::trace_environment_t&             env,
             std::optional<typename Types::node_id_t>         node_id,
             std::optional<typename Types::process_id_t>      process_id,
             std::optional<typename Types::thread_id_t>       thread_id,
             std::optional<typename Types::agent_unique_id_t> agent_id,
             std::optional<typename Types::stream_id_t>       stream_id,
             std::optional<typename Types::queue_id_t>        queue_id,
             std::optional<typename Types::track_name_t>      track_name) {
        env.node_id    = node_id;
        env.process_id = process_id;
        env.thread_id  = thread_id;
        env.agent_id   = agent_id;
        env.stream_id  = stream_id;
        env.queue_id   = queue_id;
        env.track_name = track_name;
    };

template <typename Types>
concept node_info_policy =
    requires(typename Types::node_info_t& node, typename Types::node_id_t node_id,
             std::size_t hash, std::string_view text,
             std::optional<std::string_view> optional_text) {
        node.node_id       = node_id;
        node.hash          = hash;
        node.machine_id    = text;
        node.system_name   = optional_text;
        node.hostname      = optional_text;
        node.release       = optional_text;
        node.version       = optional_text;
        node.hardware_name = optional_text;
        node.domain_name   = optional_text;
    };

template <typename Types>
concept process_info_policy =
    requires(typename Types::process_info_t& process,
             typename Types::process_id_t process_id, typename Types::node_id_t node_id,
             std::size_t number, std::string_view text,
             std::optional<std::string_view> optional_text) {
        process.ppid        = number;
        process.pid         = process_id;
        process.init        = number;
        process.fini        = number;
        process.start       = number;
        process.end         = number;
        process.command     = optional_text;
        process.environment = text;
        process.extdata     = text;
        process.node_id     = node_id;
    };

template <typename Types>
concept agent_info_policy =
    requires(typename Types::agent_info_t&     agent,
             typename Types::agent_unique_id_t unique_id,
             typename Types::node_id_t node_id, typename Types::process_id_t process_id,
             std::size_t number, std::string_view text,
             std::optional<std::string_view> optional_text) {
        agent.unique_id      = unique_id;
        agent.absolute_index = number;
        agent.logical_index  = number;
        agent.uuid           = number;
        agent.name           = optional_text;
        agent.model_name     = optional_text;
        agent.vendor_name    = optional_text;
        agent.product_name   = optional_text;
        agent.user_name      = optional_text;
        agent.extdata        = text;
        agent.node_id        = node_id;
        agent.process_id     = process_id;
    };

template <typename Types>
concept pmc_info_unique_id_policy =
    requires(typename Types::pmc_info_unique_id_t&            uid,
             typename Types::pmc_description_name_t           name,
             std::optional<typename Types::agent_unique_id_t> agent_id) {
        uid.name     = name;
        uid.agent_id = agent_id;
        { uid == uid } -> std::convertible_to<bool>;
    };

template <typename Types>
concept pmc_info_policy =
    requires(typename Types::pmc_info_t&          pmc,
             typename Types::pmc_info_unique_id_t unique_id,
             typename Types::node_id_t node_id, typename Types::process_id_t process_id,
             std::size_t number, std::string_view text,
             std::optional<std::string_view> optional_text) {
        pmc.unique_id        = unique_id;
        pmc.target_arch      = optional_text;
        pmc.event_code       = number;
        pmc.instance_id      = number;
        pmc.symbol           = text;
        pmc.description      = optional_text;
        pmc.long_description = optional_text;
        pmc.component        = optional_text;
        pmc.units            = optional_text;
        pmc.value_type       = optional_text;
        pmc.block            = optional_text;
        pmc.expression       = optional_text;
        pmc.is_constant      = number;
        pmc.is_derived       = number;
        pmc.extdata          = text;
        pmc.node_id          = node_id;
        pmc.process_id       = process_id;
    };

template <typename Types>
concept thread_info_policy =
    requires(typename Types::thread_info_t& thread, typename Types::thread_id_t thread_id,
             typename Types::node_id_t node_id, typename Types::process_id_t process_id,
             std::size_t number, std::string_view text,
             std::optional<std::string_view> optional_text) {
        thread.parent_process_id = number;
        thread.thread_id         = thread_id;
        thread.name              = optional_text;
        thread.start             = number;
        thread.end               = number;
        thread.extdata           = text;
        thread.node_id           = node_id;
        thread.process_id        = process_id;
    };

template <typename Types>
concept stream_info_policy =
    requires(typename Types::stream_info_t& stream, typename Types::stream_id_t stream_id,
             typename Types::node_id_t node_id, typename Types::process_id_t process_id,
             std::string_view text, std::optional<std::string_view> optional_text) {
        stream.stream_id  = stream_id;
        stream.name       = optional_text;
        stream.extdata    = text;
        stream.node_id    = node_id;
        stream.process_id = process_id;
    };

template <typename Types>
concept queue_info_policy =
    requires(typename Types::queue_info_t& queue, typename Types::queue_id_t queue_id,
             typename Types::node_id_t node_id, typename Types::process_id_t process_id,
             std::string_view text, std::optional<std::string_view> optional_text) {
        queue.queue_id   = queue_id;
        queue.name       = optional_text;
        queue.extdata    = text;
        queue.node_id    = node_id;
        queue.process_id = process_id;
    };

template <typename Types>
concept code_object_info_policy =
    requires(typename Types::code_object_info_t& code_object,
             typename Types::code_object_id_t    code_object_id,
             typename Types::node_id_t node_id, typename Types::process_id_t process_id,
             std::optional<typename Types::agent_unique_id_t> agent_id,
             std::size_t number, std::string_view text,
             std::optional<std::string_view> optional_text) {
        code_object.id           = code_object_id;
        code_object.uri          = optional_text;
        code_object.load_base    = number;
        code_object.load_size    = number;
        code_object.load_delta   = number;
        code_object.storage_type = optional_text;
        code_object.extdata      = text;
        code_object.node_id      = node_id;
        code_object.process_id   = process_id;
        code_object.agent_id     = agent_id;
    };

template <typename Types>
concept kernel_symbol_info_policy =
    requires(typename Types::kernel_symbol_info_t& symbol,
             typename Types::kernel_symbol_id_t    symbol_id,
             typename Types::node_id_t node_id, typename Types::process_id_t process_id,
             typename Types::code_object_id_t code_object_id, std::size_t number,
             std::string_view text, std::optional<std::string_view> optional_text) {
        symbol.id                        = symbol_id;
        symbol.name                      = optional_text;
        symbol.display_name              = optional_text;
        symbol.kernel_object             = number;
        symbol.kernarg_segment_size      = number;
        symbol.kernarg_segment_alignment = number;
        symbol.group_segment_size        = number;
        symbol.private_segment_size      = number;
        symbol.sgpr_count                = number;
        symbol.arch_vgpr_count           = number;
        symbol.accum_vgpr_count          = number;
        symbol.extdata                   = text;
        symbol.node_id                   = node_id;
        symbol.process_id                = process_id;
        symbol.code_obj_id               = code_object_id;
    };

template <typename Types>
concept track_info_policy = requires(
    typename Types::track_info_t& track, std::optional<typename Types::track_name_t> name,
    typename Types::node_id_t                   node_id,
    std::optional<typename Types::process_id_t> process_id,
    std::optional<typename Types::thread_id_t> thread_id, std::string_view text) {
    track.name       = name;
    track.extdata    = text;
    track.node_id    = node_id;
    track.process_id = process_id;
    track.thread_id  = thread_id;
    { track == track } -> std::convertible_to<bool>;
};

template <typename Types>
concept arg_data_policy =
    requires(typename Types::arg_data_t& arg, std::size_t position, std::string_view text,
             std::optional<std::string_view> optional_text) {
        arg.position = position;
        arg.type     = text;
        arg.name     = text;
        arg.value    = optional_text;
        arg.extdata  = text;
    };

template <typename Types>
concept event_data_policy =
    requires(typename Types::event_data_t&            event,
             std::optional<std::size_t>               optional_number,
             typename Types::call_stack_t             call_stack,
             typename Types::source_context_list_t    line_info,
             typename Types::call_stack_t::value_type frame, std::string_view text,
             std::optional<std::string_view> optional_text) {
        event.stack_id        = optional_number;
        event.parent_stack_id = optional_number;
        event.correlation_id  = optional_number;
        event.call_stack      = call_stack;
        event.line_info_list  = line_info;
        event.event_category  = optional_text;
        event.extdata         = text;
        event.call_stack.push_back(frame);
    };

template <typename Types>
concept region_data_policy =
    requires(typename Types::region_data_t&              region,
             std::optional<typename Types::event_data_t> event,
             typename Types::timestamp_ns_t timestamp, typename Types::arg_data_t arg,
             std::string_view text) {
        region.event           = event;
        region.start_timestamp = timestamp;
        region.end_timestamp   = timestamp;
        region.name            = text;
        region.extdata         = text;
        region.args.push_back(arg);
    };

template <typename Types>
concept sample_data_policy =
    requires(typename Types::sample_data_t& sample,
             typename Types::timestamp_ns_t timestamp, typename Types::track_info_t track,
             std::string_view text) {
        sample.timestamp = timestamp;
        sample.track     = track;
        sample.extdata   = text;
    };

template <typename Types>
concept pmc_event_data_policy =
    requires(typename Types::pmc_event_data_t&           pmc_event,
             std::optional<typename Types::event_data_t> event, double value,
             std::string_view text, typename Types::sample_data_t sample) {
        pmc_event.event   = event;
        pmc_event.value   = value;
        pmc_event.extdata = text;
        pmc_event.sample  = sample;
    };

template <typename Types>
concept kernel_dispatch_data_policy =
    requires(typename Types::kernel_dispatch_data_t&     dispatch,
             std::optional<typename Types::event_data_t> event,
             typename Types::timestamp_ns_t              timestamp,
             typename Types::kernel_symbol_id_t          symbol_id,
             typename Types::code_object_id_t code_object_id, std::size_t number,
             std::string_view text, std::optional<std::string_view> optional_text) {
        dispatch.event                = event;
        dispatch.dispatch_id          = number;
        dispatch.start_timestamp      = timestamp;
        dispatch.end_timestamp        = timestamp;
        dispatch.kernel_symbol_id     = symbol_id;
        dispatch.code_object_id       = code_object_id;
        dispatch.private_segment_size = number;
        dispatch.group_segment_size   = number;
        dispatch.workgroup_size_x     = number;
        dispatch.workgroup_size_y     = number;
        dispatch.workgroup_size_z     = number;
        dispatch.grid_size_x          = number;
        dispatch.grid_size_y          = number;
        dispatch.grid_size_z          = number;
        dispatch.name                 = optional_text;
        dispatch.extdata              = text;
    };

template <typename Types>
concept memory_copy_data_policy =
    requires(typename Types::memory_copy_data_t&              copy,
             std::optional<typename Types::event_data_t>      event,
             typename Types::timestamp_ns_t                   timestamp,
             std::optional<typename Types::agent_unique_id_t> agent_id,
             std::optional<std::size_t> optional_number, std::size_t number,
             std::string_view text, std::optional<std::string_view> optional_text) {
        copy.event           = event;
        copy.start_timestamp = timestamp;
        copy.end_timestamp   = timestamp;
        copy.dst_agent_id    = agent_id;
        copy.dst_address     = optional_number;
        copy.src_agent_id    = agent_id;
        copy.src_address     = optional_number;
        copy.size            = number;
        copy.name            = text;
        copy.region_name     = optional_text;
        copy.extdata         = text;
    };

template <typename Types>
concept memory_alloc_data_policy =
    requires(typename Types::memory_alloc_data_t&        alloc,
             std::optional<typename Types::event_data_t> event,
             typename Types::timestamp_ns_t              timestamp,
             std::optional<std::size_t> optional_number, std::size_t number,
             std::string_view text, std::optional<std::string_view> optional_text) {
        alloc.event           = event;
        alloc.type            = optional_text;
        alloc.level           = optional_text;
        alloc.start_timestamp = timestamp;
        alloc.end_timestamp   = timestamp;
        alloc.address         = optional_number;
        alloc.size            = number;
        alloc.extdata         = text;
    };

/**
 * Full set of record types a writer consumes. `Types` is a bundle of aliases; see the
 * individual `*_policy` concepts for the fields each record must expose.
 */
template <typename Types>
concept writer_types_policy =
    writer_id_policy<typename Types::node_id_t> &&
    writer_id_policy<typename Types::process_id_t> &&
    writer_id_policy<typename Types::thread_id_t> &&
    writer_id_policy<typename Types::code_object_id_t> &&
    writer_id_policy<typename Types::kernel_symbol_id_t> &&
    writer_id_policy<typename Types::stream_id_t> &&
    writer_id_policy<typename Types::queue_id_t> &&
    writer_id_policy<typename Types::timestamp_ns_t> &&
    std::convertible_to<std::string_view, typename Types::track_name_t> &&
    std::convertible_to<std::string_view, typename Types::pmc_description_name_t> &&
    std::default_initializable<typename Types::call_stack_t> &&
    std::default_initializable<typename Types::source_context_list_t> &&
    agent_unique_id_policy<Types> && trace_environment_policy<Types> &&
    node_info_policy<Types> && process_info_policy<Types> && agent_info_policy<Types> &&
    pmc_info_unique_id_policy<Types> && pmc_info_policy<Types> &&
    thread_info_policy<Types> && stream_info_policy<Types> && queue_info_policy<Types> &&
    code_object_info_policy<Types> && kernel_symbol_info_policy<Types> &&
    track_info_policy<Types> && arg_data_policy<Types> && event_data_policy<Types> &&
    region_data_policy<Types> && sample_data_policy<Types> &&
    pmc_event_data_policy<Types> && kernel_dispatch_data_policy<Types> &&
    memory_copy_data_policy<Types> && memory_alloc_data_policy<Types>;

}  // namespace rocprofsys::policies::profiler_hub
