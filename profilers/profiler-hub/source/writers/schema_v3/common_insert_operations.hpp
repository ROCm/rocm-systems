// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "writers/common_insert_operations.hpp"
#include "writers/writer_context.hpp"

#include "data_storage/schema_v3/insert_statements.hpp"
#include "data_storage/schema_version.hpp"
#include "json_serializers.hpp"
#include "profiler-hub/writer_types.hpp"

#include "debug.hpp"

#include <memory>
#include <optional>
#include <stdexcept>

namespace profiler_hub
{

template <>
class common_insert_operations<data_storage::schema_v3_tag>
{
public:
    explicit common_insert_operations(
        std::shared_ptr<writer_context> ctx,
        std::shared_ptr<
            data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
            stmts)
    : m_ctx(std::move(ctx))
    , m_stmts(std::move(stmts))
    {}

    primary_key_t insert_event(const writer_types::event_data_t& event_data)
    {
        auto& string_info_utility = m_ctx->registry->string_info();

        if(event_data.event_category.has_value() &&
           !string_info_utility.is_entry_registered(event_data.event_category.value()))
        {
            register_string(event_data.event_category.value());
        }

        const auto event_category_pk =
            event_data.event_category.has_value()
                ? std::make_optional(string_info_utility.get_primary_key_value_for_entity(
                      event_data.event_category.value()))
                : std::nullopt;
        const auto pk = m_ctx->key_providers->event_data().get_primary_key_value();

        m_stmts->event_statement()(
            pk,
            event_category_pk,
            event_data.stack_id,
            event_data.parent_stack_id,
            event_data.correlation_id,
            json_serializers::serialize_call_stack(event_data.call_stack),
            json_serializers::serialize_source_context(event_data.line_info_list),
            event_data.extdata);

        return pk;
    }

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
        const auto pk = m_ctx->key_providers->sample_data().get_primary_key_value();

        m_stmts->sample_statement()(
            pk, track_pk, sample_data.timestamp, event_pk, sample_data.extdata);
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

private:
    std::shared_ptr<writer_context> m_ctx;
    std::shared_ptr<
        data_storage::schema_v3::insert_statements<data_storage::sqlite_backend>>
        m_stmts;
};

}  // namespace profiler_hub
