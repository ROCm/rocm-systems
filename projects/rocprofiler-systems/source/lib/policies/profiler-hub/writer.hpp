// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "policies/profiler-hub/writer_types.hpp"

#include <concepts>
#include <string>
#include <string_view>

namespace rocprofsys::policies::profiler_hub
{

/**
 * Surface of a profiler-hub writer used by `rocpd_processor_t`.
 *
 * A writer is owned by value and built in place from the database path and the
 * session uuid. It exposes the record types as nested aliases (e.g.
 * `Writer::node_info_t`) checked by `writer_types_policy`. All operations are consumed
 * synchronously: record string fields are non-owning views, so implementations must not
 * retain them after returning.
 */
template <typename Writer>
concept writer_policy =
    writer_types_policy<Writer> &&
    std::constructible_from<Writer, const std::string&, const std::string&> &&
    requires(Writer& writer, const typename Writer::node_info_t& node,
             const typename Writer::process_info_t&       process,
             const typename Writer::agent_info_t&         agent,
             const typename Writer::pmc_info_t&           pmc,
             const typename Writer::thread_info_t&        thread,
             const typename Writer::stream_info_t&        stream,
             const typename Writer::queue_info_t&         queue,
             const typename Writer::code_object_info_t&   code_object,
             const typename Writer::kernel_symbol_info_t& kernel_symbol,
             const typename Writer::track_info_t& track, const std::string_view str,
             const typename Writer::region_data_t&          region,
             const typename Writer::pmc_event_data_t&       pmc_event,
             const typename Writer::pmc_info_unique_id_t&   pmc_uid,
             const typename Writer::kernel_dispatch_data_t& dispatch,
             const typename Writer::memory_copy_data_t&     copy,
             const typename Writer::memory_alloc_data_t&    alloc,
             const typename Writer::trace_environment_t&    env) {
        { writer.register_node_info(node) } -> std::same_as<void>;
        { writer.register_process_info(process) } -> std::same_as<void>;
        { writer.register_agent_info(agent) } -> std::same_as<void>;
        { writer.register_pmc_info(pmc) } -> std::same_as<void>;
        { writer.register_thread_info(thread) } -> std::same_as<void>;
        { writer.register_stream_info(stream) } -> std::same_as<void>;
        { writer.register_queue_info(queue) } -> std::same_as<void>;
        { writer.register_code_object_info(code_object) } -> std::same_as<void>;
        { writer.register_kernel_symbol_info(kernel_symbol) } -> std::same_as<void>;
        { writer.register_track_info(track) } -> std::same_as<void>;
        { writer.register_string(str) } -> std::same_as<void>;

        { writer.insert_region_data(region, env) } -> std::same_as<void>;
        { writer.insert_pmc_event_data(pmc_event, pmc_uid) } -> std::same_as<void>;
        { writer.insert_kernel_dispatch_data(dispatch, env) } -> std::same_as<void>;
        { writer.insert_memory_copy_data(copy, env) } -> std::same_as<void>;
        { writer.insert_memory_alloc_data(alloc, env) } -> std::same_as<void>;

        { writer.flush_in_memory_data_to_disk() } -> std::same_as<void>;
    };

}  // namespace rocprofsys::policies::profiler_hub
