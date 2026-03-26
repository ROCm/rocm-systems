// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "writers/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v4/insert_statements.hpp"
#include "data_storage/schema_version.hpp"
#include "rocpdsna/writer_types.hpp"
#include "rocpdsna/shared_types.hpp"

#include "debug.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace rocpdsna
{

/**
 * @brief Common insert operations for schema v4 (latest)
 *
 * Key differences from schema_v3:
 * - Event: NO call_stack or line_info columns (use separate tables)
 * - Region/dispatch/memory: Uses track_id + timestamp IDs instead of nid/pid/tid + direct timestamps
 * - Sample: Uses name_id + timestamp_id
 * - New operations: insert_timestamp(), insert_call_stack(), insert_line_info(), insert_category()
 */
template <>
class common_insert_operations<data_storage::schema_v4_tag>
{
public:
    explicit common_insert_operations(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v4::insert_statements<data_storage::sqlite_backend>>
            stmts)
    : m_ctx(std::move(ctx))
    , m_stmts(std::move(stmts))
    {}

    /**
     * @brief Insert event data (v4 schema - no call_stack/line_info columns)
     *
     * In v4, call_stack and line_info are stored in separate tables:
     * - rocpd_call_stack (event_id, pc_id, depth, extdata)
     * - rocpd_line_info (event_id, source_code_id, pc_id, extdata)
     */
    primary_key_t insert_event(const writer_types::event_data_t& event_data)
    {
        // Handle event category (stored in rocpd_info_category table in v4)
        std::optional<primary_key_t> category_pk = std::nullopt;
        if(event_data.event_category.has_value())
        {
            category_pk = ensure_category_registered(event_data.event_category.value());
        }

        const auto pk = m_ctx->key_providers->event_data().get_primary_key_value();

        // v4 event: id, category_id, stack_id, parent_stack_id, correlation_id, extdata
        // NO call_stack, NO line_info columns
        m_stmts->event_statement()(
            pk,
            category_pk,
            event_data.stack_id,
            event_data.parent_stack_id,
            event_data.correlation_id,
            event_data.extdata);

        // If we have call_stack data, insert into separate table
        if(!event_data.call_stack.empty())
        {
            insert_call_stack_entries(pk, event_data.call_stack);
        }

        // If we have line_info data, insert into separate table
        if(!event_data.line_info_list.empty())
        {
            insert_line_info_entries(pk, event_data.line_info_list);
        }

        return pk;
    }

    /**
     * @brief Insert timestamp into rocpd_timestamp table
     *
     * @param value The timestamp value
     * @param phase Optional phase indicator (0 = begin, 1 = end, etc.)
     * @param track_id Optional track ID reference
     * @return Primary key of the inserted timestamp
     */
    primary_key_t insert_timestamp(uint64_t value,
                                   std::optional<int> phase = std::nullopt,
                                   std::optional<primary_key_t> track_id = std::nullopt)
    {
        const auto pk = m_ctx->key_providers->timestamp_data().get_primary_key_value();
        m_stmts->timestamp_statement()(pk, value, phase, track_id);
        return pk;
    }

    /**
     * @brief Ensure category is registered and return its primary key
     */
    primary_key_t ensure_category_registered(std::string_view category_name)
    {
        auto it = m_category_cache.find(std::string(category_name));
        if(it != m_category_cache.end())
        {
            return it->second;
        }

        const auto pk = m_ctx->key_providers->category_info().get_primary_key_value();
        m_stmts->category_info_statement()(pk, category_name, "{}");
        m_category_cache[std::string(category_name)] = pk;
        return pk;
    }

    /**
     * @brief Insert sample data (v4 schema - uses name_id, timestamp_id)
     */
    void insert_sample(const writer_types::sample_data_t& sample_data,
                       const primary_key_t&               event_pk)
    {
        auto& track_info_utility = m_ctx->registry->track_info();

        if(!track_info_utility.is_entry_registered(sample_data.track))
        {
            const auto* const track_name_print_value =
                sample_data.track.name.has_value() ? sample_data.track.name.value().data()
                                                   : "[NULL]";

            throw std::runtime_error(
                fmt::format("Track not registered for Sample Data: track_name: {} "
                            "node_id: {} process_id: {} thread_id: {}",
                            track_name_print_value,
                            sample_data.track.node_id,
                            sample_data.track.process_id.has_value()
                                ? std::to_string(sample_data.track.process_id.value())
                                : "[NULL]",
                            sample_data.track.thread_id.has_value()
                                ? std::to_string(sample_data.track.thread_id.value())
                                : "[NULL]"));
        }

        const auto track_pk =
            track_info_utility.get_primary_key_value_for_entity(sample_data.track);

        // Register track name as string and get name_id
        std::string_view track_name_view = sample_data.track.name.has_value()
            ? sample_data.track.name.value()
            : "";
        ensure_string_registered(track_name_view);
        const auto name_pk = m_ctx->registry->string_info().get_primary_key_value_for_entity(
            std::string(track_name_view));

        // Insert timestamp and get ID
        const auto timestamp_id = insert_timestamp(sample_data.timestamp);

        const auto pk = m_ctx->key_providers->sample_data().get_primary_key_value();

        // v4 sample: id, track_id, name_id, timestamp_id, event_id, extdata
        m_stmts->sample_statement()(
            pk, track_pk, name_pk, timestamp_id, event_pk, sample_data.extdata);
    }

    void insert_arg(const writer_types::arg_data_t& arg_data, primary_key_t event_id)
    {
        if(arg_data.type.empty() || arg_data.name.empty())
        {
            throw std::runtime_error(
                fmt::format("Type or name is empty for Arg Data: type: {}, name: {}",
                            arg_data.type,
                            arg_data.name));
        }

        const auto pk = m_ctx->key_providers->arg().get_primary_key_value();

        m_stmts->arg_statement()(pk,
                                 event_id,
                                 arg_data.position,
                                 arg_data.type,
                                 arg_data.name,
                                 arg_data.value,
                                 arg_data.extdata);
    }

    /**
     * @brief Insert or get string and return its primary key
     */
    primary_key_t insert_string(std::string_view str)
    {
        if(str.empty())
        {
            throw std::runtime_error("Trying to insert empty string");
        }

        auto& string_info_utility = m_ctx->registry->string_info();

        if(string_info_utility.is_entry_registered(str))
        {
            return string_info_utility.get_primary_key_value_for_entity(str);
        }

        const std::string str_owned(str);
        const auto pk = m_ctx->key_providers->string_info().get_primary_key_value();
        m_stmts->string_statement()(pk, str);
        string_info_utility.emplace_entity(str_owned, pk);

        LOG_TRACE("Inserted string: {}", str);
        return pk;
    }

    void register_string(std::string_view str)
    {
        if(str.empty())
        {
            throw std::runtime_error("Trying to register empty string");
        }

        auto& string_info_utility = m_ctx->registry->string_info();

        if(string_info_utility.is_entry_registered(str))
        {
            LOG_WARNING("String already registered: str: {}", str);
            return;
        }

        const std::string str_owned(str);
        const auto pk = m_ctx->key_providers->string_info().get_primary_key_value();
        m_stmts->string_statement()(pk, str);
        string_info_utility.emplace_entity(str_owned, pk);

        LOG_TRACE("Registered string: {}", str);
    }

    void ensure_string_registered(std::string_view str)
    {
        if(str.empty())
        {
            return;
        }
        auto& string_info_utility = m_ctx->registry->string_info();
        if(!string_info_utility.is_entry_registered(str))
        {
            register_string(str);
        }
    }

    void ensure_optional_string_registered(std::optional<std::string_view> str)
    {
        if(str.has_value())
        {
            ensure_string_registered(str.value());
        }
    }

    void maybe_insert_sample(const writer_types::trace_environment_t& trace_env,
                             uint64_t                                 timestamp,
                             std::optional<primary_key_t>             event_pk)
    {
        if(trace_env.track_name.has_value() && event_pk.has_value())
        {
            const writer_types::track_info_t  track_info = { trace_env.track_name.value(),
                                                             "{}",
                                                             trace_env.node_id.value(),
                                                             trace_env.process_id.value(),
                                                             trace_env.thread_id.value() };
            const writer_types::sample_data_t sample_data = { timestamp,
                                                              track_info,
                                                              "{}" };
            insert_sample(sample_data, event_pk.value());
        }
    }

    /**
     * @brief Get or create track ID for given trace environment
     *
     * In v4, all data tables use track_id instead of nid/pid/tid columns.
     * This method resolves or creates the appropriate track, auto-registering
     * if the track doesn't exist.
     */
    primary_key_t resolve_track_id(const writer_types::trace_environment_t& trace_env)
    {
        auto& track_info_utility = m_ctx->registry->track_info();

        // Build a track_info from trace_environment
        writer_types::track_info_t track_info;
        track_info.name = trace_env.track_name;
        track_info.extdata = "{}";
        track_info.node_id = trace_env.node_id.value();
        if(trace_env.process_id.has_value())
        {
            track_info.process_id = trace_env.process_id.value();
        }
        if(trace_env.thread_id.has_value())
        {
            track_info.thread_id = trace_env.thread_id.value();
        }

        // Auto-register track if it doesn't exist (similar to v3 behavior)
        if(!track_info_utility.is_entry_registered(track_info))
        {
            // Get or create required foreign keys
            const auto nid = trace_env.node_id.value();

            std::optional<primary_key_t> pid_fk = std::nullopt;
            if(trace_env.process_id.has_value())
            {
                pid_fk = m_ctx->validator->resolve_process_key(trace_env.process_id);
            }

            std::optional<primary_key_t> tid_fk = std::nullopt;
            if(trace_env.thread_id.has_value())
            {
                tid_fk = m_ctx->validator->resolve_optional_thread_key(trace_env.thread_id);
            }

            std::optional<primary_key_t> agent_fk = std::nullopt;
            if(trace_env.agent_id.has_value())
            {
                agent_fk = m_ctx->validator->resolve_optional_agent_key(trace_env.agent_id);
            }

            std::optional<primary_key_t> queue_fk = std::nullopt;
            if(trace_env.queue_id.has_value())
            {
                queue_fk = m_ctx->validator->resolve_optional_queue_key(trace_env.queue_id);
            }

            std::optional<primary_key_t> stream_fk = std::nullopt;
            if(trace_env.stream_id.has_value())
            {
                stream_fk = m_ctx->validator->resolve_optional_stream_key(trace_env.stream_id);
            }

            // name_id is optional - create one if track has a name
            std::optional<primary_key_t> name_fk = std::nullopt;
            if(trace_env.track_name.has_value() && !trace_env.track_name.value().empty())
            {
                name_fk = insert_string(trace_env.track_name.value());
            }

            const auto track_pk = m_ctx->key_providers->track_info().get_primary_key_value();

            // rocpd_track: id, nid, ppid, pid, tid, agent_id, queue_id, stream_id, name_id, extdata
            // ppid (parent process id) is not in trace_env, so set to nullopt
            m_stmts->track_info_statement()(track_pk,
                                            nid,
                                            std::nullopt,  // ppid
                                            pid_fk,
                                            tid_fk,
                                            agent_fk,
                                            queue_fk,
                                            stream_fk,
                                            name_fk,
                                            std::string_view("{}"));

            track_info_utility.emplace_entity(track_info, track_pk);
            LOG_TRACE("Auto-registered track: nid={}, pid={}, tid={}",
                      nid,
                      trace_env.process_id.has_value() ? std::to_string(trace_env.process_id.value()) : "[NULL]",
                      trace_env.thread_id.has_value() ? std::to_string(trace_env.thread_id.value()) : "[NULL]");
        }

        return track_info_utility.get_primary_key_value_for_entity(track_info);
    }

private:
    /**
     * @brief Insert call stack entries into rocpd_call_stack table
     */
    void insert_call_stack_entries(primary_key_t event_id,
                                   const shared_types::call_stack_t& entries)
    {
        size_t depth = 0;
        for(const auto& entry : entries)
        {
            const auto pk = m_ctx->key_providers->call_stack_data().get_primary_key_value();
            // rocpd_call_stack: id, event_id, pc_id, depth, extdata
            // pc_id is optional - we might need to create/reference a pc entry
            m_stmts->call_stack_statement()(pk, event_id, std::nullopt, depth, entry.extdata);
            ++depth;
        }
    }

    /**
     * @brief Insert line info entries into rocpd_line_info table
     */
    void insert_line_info_entries(primary_key_t event_id,
                                  const shared_types::source_context_list_t& entries)
    {
        for(const auto& entry : entries)
        {
            const auto pk = m_ctx->key_providers->line_info_data().get_primary_key_value();
            // rocpd_line_info: id, event_id, source_code_id, pc_id, extdata
            // source_code_id and pc_id are optional
            m_stmts->line_info_statement()(pk, event_id, std::nullopt, std::nullopt, "{}");
        }
    }

    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v4::insert_statements<data_storage::sqlite_backend>>
        m_stmts;

    // Cache for category IDs to avoid duplicate inserts
    std::unordered_map<std::string, primary_key_t> m_category_cache;
};

}  // namespace rocpdsna
