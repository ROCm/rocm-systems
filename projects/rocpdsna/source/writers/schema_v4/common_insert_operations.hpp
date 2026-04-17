// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "writers/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v4/insert_statements.hpp"
#include "data_storage/schema_version.hpp"
#include "rocpdsna/shared_types.hpp"
#include "rocpdsna/writer_types.hpp"

#include "debug.hpp"

#include <nlohmann/json.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocpdsna
{

/**
 * @brief Common insert operations for schema v4 (latest)
 *
 * Key differences from schema_v3:
 * - Event: NO call_stack or line_info columns (use separate tables)
 * - Region/dispatch/memory: Uses track_id + timestamp IDs instead of nid/pid/tid + direct
 * timestamps
 * - Sample: Uses name_id + timestamp_id
 * - New operations: insert_timestamp(), insert_call_stack(), insert_line_info(),
 * insert_category()
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
     *
     * When call_stack / line_info embed address_range, source_code, or program_counter
     * data, those structs are materialized into rocpd_info_address_range,
     * rocpd_info_source_code, and rocpd_info_pc. That requires trace context:
     * pass logical node_id and process_id (same as register_* info APIs).
     */
    primary_key_t insert_event(
        const writer_types::event_data_t&         event_data,
        std::optional<writer_types::node_id_t>    ctx_node_id    = std::nullopt,
        std::optional<writer_types::process_id_t> ctx_process_id = std::nullopt)
    {
        if(event_requires_trace_for_embedded_info(event_data))
        {
            if(!ctx_node_id.has_value() || !ctx_process_id.has_value())
            {
                throw std::runtime_error(
                    "insert_event (schema v4): node_id and process_id are required when "
                    "event_data contains call_stack or line_info entries with "
                    "address_range, source_code, or program_counter (info rows must be "
                    "inserted with nid/pid FKs)");
            }
            m_ctx->validator->require_node(*ctx_node_id).require_process(*ctx_process_id);
        }

        std::optional<primary_key_t> process_pk;
        if(ctx_node_id.has_value() && ctx_process_id.has_value())
        {
            process_pk = m_ctx->validator->resolve_process_key(ctx_process_id.value());
        }

        // Handle event category (stored in rocpd_info_category table in v4)
        std::optional<primary_key_t> category_pk = std::nullopt;
        if(event_data.event_category.has_value())
        {
            category_pk = ensure_category_registered(event_data.event_category.value());
        }

        const auto pk = m_ctx->key_providers->event_data().get_primary_key_value();

        // v4 event: id, category_id, stack_id, parent_stack_id, correlation_id, extdata
        // NO call_stack, NO line_info columns
        m_stmts->event_statement()(pk,
                                   category_pk,
                                   event_data.stack_id,
                                   event_data.parent_stack_id,
                                   event_data.correlation_id,
                                   event_data.extdata);

        // If we have call_stack data, insert into separate table
        if(!event_data.call_stack.empty())
        {
            insert_call_stack_entries(pk, event_data.call_stack, ctx_node_id, process_pk);
        }

        // If we have line_info data, insert into separate table
        if(!event_data.line_info_list.empty())
        {
            insert_line_info_entries(
                pk, event_data.line_info_list, ctx_node_id, process_pk);
        }

        return pk;
    }

    /**
     * @brief Insert timestamp into rocpd_timestamp table
     *
     * @param value The timestamp value
     * @param phase Optional phase indicator (0 = none/instantaneous, 1 =
     * start/enter/load, 2 = end/exit/unload)
     * @param track_id Optional track ID reference
     * @return Primary key of the inserted timestamp
     */
    primary_key_t insert_timestamp(uint64_t                     value,
                                   std::optional<int>           phase    = std::nullopt,
                                   std::optional<primary_key_t> track_id = std::nullopt)
    {
        const auto pk = m_ctx->key_providers->timestamp_data().get_primary_key_value();
        m_stmts->timestamp_statement()(pk, value, phase, track_id);
        return pk;
    }

    /**
     * @brief Ensure category is registered and return its primary key
     *
     * Uses the category_info registry (similar to how v3 uses string_info for categories)
     */
    primary_key_t ensure_category_registered(
        std::string_view category_name,
        std::string_view extdata = writer_types::empty_json)
    {
        auto& category_info_utility = m_ctx->registry->category_info();

        if(category_info_utility.is_entry_registered(std::string(category_name)))
        {
            return category_info_utility.get_primary_key_value_for_entity(
                std::string(category_name));
        }

        const auto pk = m_ctx->key_providers->category_info().get_primary_key_value();
        m_stmts->category_info_statement()(pk, category_name, extdata);
        category_info_utility.emplace_entity(std::string(category_name), pk);
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
        std::string_view track_name_view =
            sample_data.track.name.has_value() ? sample_data.track.name.value() : "";
        ensure_string_registered(track_name_view);
        const auto name_pk =
            m_ctx->registry->string_info().get_primary_key_value_for_entity(
                std::string(track_name_view));

        // Insert timestamp and get ID
        // Phase 0 = none/instantaneous for sample timestamps
        const auto timestamp_id = insert_timestamp(sample_data.timestamp, 0, track_pk);

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
                             std::optional<primary_key_t>             event_pk,
                             std::string_view sample_extdata = writer_types::empty_json,
                             std::string_view track_extdata  = writer_types::empty_json)
    {
        if(trace_env.track_name.has_value() && event_pk.has_value())
        {
            writer_types::track_info_t track_info;
            track_info.name       = trace_env.track_name;
            track_info.extdata    = track_extdata;
            track_info.node_id    = trace_env.node_id.value();
            track_info.process_id = trace_env.process_id;
            track_info.thread_id  = trace_env.thread_id;
            track_info.ppid       = trace_env.ppid;
            track_info.agent_id   = trace_env.agent_id;
            track_info.queue_id   = trace_env.queue_id;
            track_info.stream_id  = trace_env.stream_id;

            writer_types::sample_data_t sample_data;
            sample_data.timestamp = timestamp;
            sample_data.track     = track_info;
            sample_data.extdata   = sample_extdata;
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
    primary_key_t resolve_track_id(
        const writer_types::trace_environment_t& trace_env,
        std::string_view                         track_extdata = writer_types::empty_json)
    {
        auto& track_info_utility = m_ctx->registry->track_info();

        // Build a track_info from trace_environment
        writer_types::track_info_t track_info;
        track_info.name    = trace_env.track_name;
        track_info.extdata = track_extdata;
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
                tid_fk =
                    m_ctx->validator->resolve_optional_thread_key(trace_env.thread_id);
            }

            std::optional<primary_key_t> agent_fk = std::nullopt;
            if(trace_env.agent_id.has_value())
            {
                agent_fk =
                    m_ctx->validator->resolve_optional_agent_key(trace_env.agent_id);
            }

            std::optional<primary_key_t> queue_fk = std::nullopt;
            if(trace_env.queue_id.has_value())
            {
                queue_fk =
                    m_ctx->validator->resolve_optional_queue_key(trace_env.queue_id);
            }

            std::optional<primary_key_t> stream_fk = std::nullopt;
            if(trace_env.stream_id.has_value())
            {
                stream_fk =
                    m_ctx->validator->resolve_optional_stream_key(trace_env.stream_id);
            }

            // name_id is optional - create one if track has a name
            std::optional<primary_key_t> name_fk = std::nullopt;
            if(trace_env.track_name.has_value() && !trace_env.track_name.value().empty())
            {
                name_fk = insert_string(trace_env.track_name.value());
            }

            const auto track_pk =
                m_ctx->key_providers->track_info().get_primary_key_value();

            std::optional<primary_key_t> ppid_fk = std::nullopt;
            if(trace_env.ppid.has_value())
            {
                ppid_fk = static_cast<primary_key_t>(trace_env.ppid.value());
            }

            m_stmts->track_info_statement()(track_pk,
                                            nid,
                                            ppid_fk,
                                            pid_fk,
                                            tid_fk,
                                            agent_fk,
                                            queue_fk,
                                            stream_fk,
                                            name_fk,
                                            track_extdata);

            track_info_utility.emplace_entity(track_info, track_pk);
            LOG_TRACE("Auto-registered track: nid={}, pid={}, tid={}",
                      nid,
                      trace_env.process_id.has_value()
                          ? std::to_string(trace_env.process_id.value())
                          : "[NULL]",
                      trace_env.thread_id.has_value()
                          ? std::to_string(trace_env.thread_id.value())
                          : "[NULL]");
        }

        return track_info_utility.get_primary_key_value_for_entity(track_info);
    }

private:
    [[nodiscard]] static bool event_requires_trace_for_embedded_info(
        const writer_types::event_data_t& event_data)
    {
        for(const auto& li : event_data.line_info_list)
        {
            if(li.source_code.has_value() || li.program_counter.has_value() ||
               li.address_range.has_value())
            {
                return true;
            }
        }
        for(const auto& fr : event_data.call_stack)
        {
            if(fr.program_counter.has_value() || fr.address_range.has_value())
            {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] static std::string json_string_view_vector(
        const std::vector<std::string_view>& lines)
    {
        nlohmann::json arr = nlohmann::json::array();
        for(const auto& line : lines)
        {
            arr.push_back(std::string(line));
        }
        return arr.dump();
    }

    [[nodiscard]] static std::string_view extdata_or_empty_json(std::string_view extdata)
    {
        return extdata.empty() ? std::string_view("{}") : extdata;
    }

    primary_key_t insert_embedded_address_range(
        writer_types::node_id_t                   nid,
        primary_key_t                             process_pk,
        const shared_types::address_range_info_t& addr)
    {
        const auto id =
            m_ctx->key_providers->address_range_info().get_primary_key_value();
        m_stmts->address_range_info_statement()(id,
                                                nid,
                                                process_pk,
                                                addr.address_base,
                                                addr.address_low,
                                                addr.address_high,
                                                extdata_or_empty_json(addr.extdata));
        return id;
    }

    primary_key_t insert_embedded_source_code(writer_types::node_id_t      nid,
                                              primary_key_t                process_pk,
                                              std::optional<primary_key_t> address_id,
                                              const shared_types::source_code_info_t& src)
    {
        const auto id = m_ctx->key_providers->source_code_info().get_primary_key_value();
        const std::string lines_json = json_string_view_vector(src.source_code_lines);
        const std::string instr_json =
            json_string_view_vector(src.assembly_instruction_lines);
        m_stmts->source_code_info_statement()(id,
                                              nid,
                                              process_pk,
                                              address_id,
                                              src.filename,
                                              src.starting_line_number,
                                              std::string_view(lines_json),
                                              std::string_view(instr_json),
                                              extdata_or_empty_json(src.extdata));
        return id;
    }

    primary_key_t insert_embedded_pc(writer_types::node_id_t      nid,
                                     primary_key_t                process_pk,
                                     std::optional<primary_key_t> address_id,
                                     const shared_types::program_counter_info_t& pc)
    {
        const auto id = m_ctx->key_providers->pc_info().get_primary_key_value();
        const std::string_view function_sv =
            (pc.function.has_value() && !pc.function->empty())
                ? pc.function.value()
                : std::string_view("<unknown>");
        m_stmts->pc_info_statement()(id,
                                     nid,
                                     process_pk,
                                     function_sv,
                                     address_id,
                                     pc.filename,
                                     pc.line_number,
                                     extdata_or_empty_json(pc.extdata));
        return id;
    }

    /**
     * @brief Insert call stack entries into rocpd_call_stack table
     */
    void insert_call_stack_entries(primary_key_t                          event_id,
                                   const shared_types::call_stack_t&      entries,
                                   std::optional<writer_types::node_id_t> ctx_node_id,
                                   std::optional<primary_key_t>           process_pk)
    {
        const bool materialize = ctx_node_id.has_value() && process_pk.has_value();
        size_t     depth       = 0;
        for(const auto& entry : entries)
        {
            const auto row_pk =
                m_ctx->key_providers->call_stack_data().get_primary_key_value();
            std::optional<primary_key_t> pc_id = std::nullopt;
            if(materialize &&
               (entry.address_range.has_value() || entry.program_counter.has_value()))
            {
                std::optional<primary_key_t> addr_pk = std::nullopt;
                if(entry.address_range.has_value())
                {
                    addr_pk = insert_embedded_address_range(ctx_node_id.value(),
                                                            process_pk.value(),
                                                            entry.address_range.value());
                }
                if(entry.program_counter.has_value())
                {
                    pc_id = insert_embedded_pc(ctx_node_id.value(),
                                               process_pk.value(),
                                               addr_pk,
                                               entry.program_counter.value());
                }
            }
            m_stmts->call_stack_statement()(
                row_pk, event_id, pc_id, depth, extdata_or_empty_json(entry.extdata));
            ++depth;
        }
    }

    /**
     * @brief Insert line info entries into rocpd_line_info table
     */
    void insert_line_info_entries(primary_key_t                              event_id,
                                  const shared_types::source_context_list_t& entries,
                                  std::optional<writer_types::node_id_t>     ctx_node_id,
                                  std::optional<primary_key_t>               process_pk)
    {
        const bool materialize = ctx_node_id.has_value() && process_pk.has_value();
        for(const auto& entry : entries)
        {
            const auto row_pk =
                m_ctx->key_providers->line_info_data().get_primary_key_value();
            std::optional<primary_key_t> source_code_id = std::nullopt;
            std::optional<primary_key_t> pc_id          = std::nullopt;
            if(materialize &&
               (entry.address_range.has_value() || entry.source_code.has_value() ||
                entry.program_counter.has_value()))
            {
                std::optional<primary_key_t> addr_pk = std::nullopt;
                if(entry.address_range.has_value())
                {
                    addr_pk = insert_embedded_address_range(ctx_node_id.value(),
                                                            process_pk.value(),
                                                            entry.address_range.value());
                }
                if(entry.source_code.has_value())
                {
                    source_code_id =
                        insert_embedded_source_code(ctx_node_id.value(),
                                                    process_pk.value(),
                                                    addr_pk,
                                                    entry.source_code.value());
                }
                if(entry.program_counter.has_value())
                {
                    pc_id = insert_embedded_pc(ctx_node_id.value(),
                                               process_pk.value(),
                                               addr_pk,
                                               entry.program_counter.value());
                }
            }
            m_stmts->line_info_statement()(row_pk,
                                           event_id,
                                           source_code_id,
                                           pc_id,
                                           extdata_or_empty_json(entry.extdata));
        }
    }

    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v4::insert_statements<data_storage::sqlite_backend>>
        m_stmts;
};

}  // namespace rocpdsna
