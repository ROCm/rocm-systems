// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/record.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/counter_samples.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "logger/debug.hpp"
#include <exception>
#include <optional>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace rocprofsys::trace_cache
{
/** The counter_column records of a schema are stored with its first batch, because the
 * storage may not be running when the sources are described. A record the storage rejects
 * is dropped and counted. */
class collection_sink final : public collection::sink_interface
{
public:
    collection::schema_id register_schema(collection::source_id     source,
                                          const collection::schema& stream) override
    {
        auto entry    = std::make_unique<stream_entry>();
        entry->source = source;
        entry->stream = stream;
        register_metadata(*entry);

        const std::lock_guard lock(m_register_mutex);
        const auto            stream_id = m_count.load(std::memory_order_relaxed);
        if(stream_id >= k_max_streams)
        {
            throw std::length_error("too many collection streams");
        }
        m_streams[stream_id] = std::move(entry);
        m_count.store(stream_id + 1, std::memory_order_release);
        return static_cast<collection::schema_id>(stream_id);
    }

    void write_batch(const collection::sample_batch& batch) noexcept override
    {
        try
        {
            auto& entry = find(batch.schema);
            std::call_once(entry.columns_stored,
                           [&] { store_columns(entry, batch.schema); });
            store_batch(entry, batch);
        } catch(const std::exception& error)
        {
            count_drop(error.what());
        }
    }

    void write_gap(const collection::gap_marker& gap) noexcept override
    {
        try
        {
            get_buffer_storage().store(gap_marker{
                .source = gap.source, .begin_ns = gap.begin_ns, .end_ns = gap.end_ns });
        } catch(const std::exception& error)
        {
            count_drop(error.what());
        }
    }

private:
    static constexpr std::size_t k_max_streams = 1024;

    struct stream_entry
    {
        collection::source_id source = 0;
        collection::schema    stream;
        std::once_flag        columns_stored;
    };

    static void register_metadata(const stream_entry& entry)
    {
        auto& registry = get_metadata_registry();
        for(std::size_t idx = 0; idx < entry.stream.columns.size(); ++idx)
        {
            const auto& column = entry.stream.columns[idx];
            registry.add_pmc_info({ .type             = entry.stream.agent,
                                    .agent_type_index = entry.stream.device,
                                    .target_arch      = entry.stream.target_arch,
                                    .event_code       = 0,
                                    .instance_id      = 0,
                                    .name             = column.pmc_name,
                                    .symbol           = column.name,
                                    .description      = column.description,
                                    .long_description = {},
                                    .component        = {},
                                    .units            = column.unit,
                                    .value_type       = k_absolute,
                                    .block            = column.block,
                                    .expression       = column.expression,
                                    .is_constant      = column.is_constant ? 1U : 0U,
                                    .is_derived       = column.is_derived ? 1U : 0U,
                                    .extdata          = "{}" });
            registry.ensure_track(column.rocpd_track_name, std::nullopt);
        }
    }

    static void store_columns(const stream_entry& entry, collection::schema_id stream_id)
    {
        for(std::uint32_t idx = 0; idx < entry.stream.columns.size(); ++idx)
        {
            const auto& column = entry.stream.columns[idx];
            get_buffer_storage().store(
                counter_column{ .source              = entry.source,
                                .schema_id           = stream_id,
                                .column              = idx,
                                .category_id         = column.category_id,
                                .agent               = entry.stream.agent,
                                .agent_index         = entry.stream.device,
                                .pmc_name            = column.pmc_name,
                                .units               = column.unit,
                                .perfetto_track_name = column.perfetto_track_name,
                                .rocpd_track_name    = column.rocpd_track_name });
        }
    }

    static void store_batch(const stream_entry&             entry,
                            const collection::sample_batch& batch)
    {
        if(batch.values.size() != entry.stream.columns.size())
        {
            throw std::invalid_argument("values do not match the columns of the schema");
        }
        get_buffer_storage().store(counter_batch_view{ .schema_id = batch.schema,
                                                       .timestamp = batch.timestamp,
                                                       .values    = batch.values });
    }

    stream_entry& find(collection::schema_id stream_id)
    {
        if(stream_id >= m_count.load(std::memory_order_acquire))
        {
            throw std::out_of_range("unknown collection stream");
        }
        return *m_streams[stream_id];
    }

    void count_drop(const char* reason) noexcept
    {
        if(m_dropped++ == 0)
        {
            LOG_WARNING("collection: record dropped, further drops are not reported: {}",
                        reason);
        }
    }

    std::mutex                                               m_register_mutex;
    std::array<std::unique_ptr<stream_entry>, k_max_streams> m_streams;
    std::atomic<std::size_t>                                 m_count{ 0 };
    std::atomic<std::uint64_t>                               m_dropped{ 0 };
};
}  // namespace rocprofsys::trace_cache
