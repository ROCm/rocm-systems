// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/sample_batch.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/trace_cache/counter_samples.hpp"
#include "logger/debug.hpp"
#include "policies/trace_cache/buffer_storage.hpp"
#include "policies/trace_cache/metadata_registry.hpp"

#include <atomic>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace rocprofsys::trace_cache
{
/**
 * Production sink: schemas go to the metadata registry, rows and gaps to the buffer
 * storage. Thread-safe; keeps no per-schema state.
 */
template <typename Storage, typename Registry>
    requires policies::trace_cache::record_storage<Storage, counter_batch_view,
                                                   gap_marker> &&
             policies::trace_cache::counter_schema_registry<Registry, collection::schema>
class collection_sink final : public collection::sink_interface
{
public:
    collection_sink(Storage& storage, Registry& registry)
    : m_storage{ storage }
    , m_registry{ registry }
    {}

    /** Logs the total drop count once if any record was dropped. */
    ~collection_sink() override
    {
        const auto dropped_count = dropped();
        if(dropped_count > 0)
        {
            try_log([dropped_count] {
                LOG_WARNING("collection: {} record(s) dropped", dropped_count);
            });
        }
    }

    collection_sink(const collection_sink&)            = delete;
    collection_sink& operator=(const collection_sink&) = delete;
    collection_sink(collection_sink&&)                 = delete;
    collection_sink& operator=(collection_sink&&)      = delete;

    [[nodiscard]] collection::schema_id register_schema(
        const collection::schema& description) override
    {
        if(description.columns.empty())
        {
            throw std::invalid_argument("collection: a schema needs at least one column");
        }
        return m_registry.add_counter_schema(description);
    }

    void write_batch(const collection::sample_batch& batch) noexcept override
    {
        store(counter_batch_view{ .schema_id = batch.schema,
                                  .timestamp = batch.timestamp,
                                  .values    = batch.values });
    }

    void write_gap(std::uint64_t begin_ns, std::uint64_t end_ns) noexcept override
    {
        store(gap_marker{ .begin_ns = begin_ns, .end_ns = end_ns });
    }

    /** Number of records the storage rejected or could not take. */
    [[nodiscard]] std::uint64_t dropped() const noexcept
    {
        return m_dropped.load(std::memory_order_relaxed);
    }

private:
    template <typename Record>
    void store(const Record& record) noexcept
    {
        // checked first so a stopped storage costs no throw/catch per record
        if(!m_storage.is_running())
        {
            count_drop("storage not running");
            return;
        }
        try
        {
            m_storage.store(record);
        } catch(const std::exception& e)
        {
            count_drop(e.what());
        } catch(...)
        {
            count_drop("unknown error");
        }
    }

    void count_drop(std::string_view reason) noexcept
    {
        m_dropped.fetch_add(1, std::memory_order_relaxed);
        if(m_warned.exchange(true, std::memory_order_relaxed))
        {
            return;
        }
        const bool logged = try_log([reason] {
            LOG_WARNING("collection: record dropped, further drops are only counted: {}",
                        reason);
        });
        if(!logged)
        {
            // warn at the next drop instead
            m_warned.store(false, std::memory_order_relaxed);
        }
    }

    // Logging runs on writer threads and in the destructor, where it must not throw.
    template <typename LogFn>
    static bool try_log(LogFn&& log) noexcept
    {
        try
        {
            std::forward<LogFn>(log)();
            return true;
        } catch(...)
        {
            return false;
        }
    }

    Storage&                   m_storage;
    Registry&                  m_registry;
    std::atomic<std::uint64_t> m_dropped{ 0 };
    std::atomic<bool>          m_warned{ false };
};
}  // namespace rocprofsys::trace_cache
