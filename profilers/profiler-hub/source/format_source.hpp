// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "connection_source.hpp"
#include "event_query.hpp"
#include "histogram_grid.hpp"
#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/reader_types.hpp"
#include "reader_catalog.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct ph_table_event;

namespace profiler_hub
{
namespace common
{
class thread_pool;
}

/** @brief What a registry entry needs to open one trace file. */
struct source_request_t
{
    std::string                  path;
    std::string                  uuid;
    reader_types::trace_format_t format;
};

/** @brief Where a format may keep what it derives from a file, and which file of the
 *         context this source is. */
struct catalog_cache_t
{
    std::string              root;
    std::vector<std::string> paths;
    uint32_t                 index{ 0 };
};

/**
 * @brief One open trace file, as ph_ctx sees it.
 *
 * A registry entry opens it. ph_ctx asks it for everything it needs from the file
 * and knows nothing of how the format stores it.
 *
 * A format must answer the catalog, track reads, which track an event is on, and
 * the histogram; an empty answer is allowed. It may answer the per-event reads
 * with nothing. It may decline the name-duration summary and the table queries:
 * ph_ctx asks first and does not call a read the format declined. A declined read
 * that is called anyway is a ph_ctx bug and throws std::logic_error.
 */
class format_source
{
public:
    explicit format_source(std::string format)
    : m_format{ std::move(format) }
    {}
    virtual ~format_source() = default;

    format_source(const format_source&)            = delete;
    format_source& operator=(const format_source&) = delete;
    format_source(format_source&&)                 = delete;
    format_source& operator=(format_source&&)      = delete;

    /** @brief The catalog this source fills. ph_ctx keeps a copy of the pointer. */
    [[nodiscard]] virtual const std::shared_ptr<reader_catalog_t>& catalog() const = 0;

    /** @brief Fills the catalog. @p workers runs any parallel part. */
    virtual void load_catalog(common::thread_pool&   workers,
                              const catalog_cache_t& cache) = 0;

    /** @brief Called once every track is loaded. A format that parsed its file into
     *         memory can free the rows here. */
    virtual void finish_load() {}

    /** @brief Whole-track reads, through the interface the track cache already takes. */
    [[nodiscard]] virtual connection_source& track_reads() = 0;

    /** @brief The track an event is on; null when the file has no such event. */
    [[nodiscard]] virtual reader_types::track_info_ptr_t track_of(
        const reader_types::timeline_event_t& event) = 0;

    [[nodiscard]] virtual reader_types::timeline_event_list_t correlated_events(
        const reader_types::timeline_event_t& /*event*/)
    {
        return {};
    }
    [[nodiscard]] virtual reader_types::call_stack_t call_stack(
        const reader_types::timeline_event_t& /*event*/)
    {
        return {};
    }
    [[nodiscard]] virtual std::optional<reader_types::region_data_t> region_details(
        const reader_types::timeline_event_t& /*event*/)
    {
        return {};
    }
    [[nodiscard]] virtual std::optional<reader_types::kernel_dispatch_data_t>
    kernel_dispatch_details(const reader_types::timeline_event_t& /*event*/)
    {
        return {};
    }
    [[nodiscard]] virtual std::optional<reader_types::memory_copy_data_t>
    memory_copy_details(const reader_types::timeline_event_t& /*event*/)
    {
        return {};
    }
    [[nodiscard]] virtual std::optional<reader_types::memory_alloc_data_t>
    memory_alloc_details(const reader_types::timeline_event_t& /*event*/)
    {
        return {};
    }

    /** @brief False declines the summary, as the reader's own summary does. */
    [[nodiscard]] virtual bool summarize_name_durations(
        const resolved_event_query& /*query*/,
        uint32_t /*event_type*/,
        std::vector<name_duration_summary_t>& /*rows*/)
    {
        return false;
    }

    /** @brief False declines read_table_events and table_event_id_span. */
    [[nodiscard]] virtual bool supports_table_events() const { return false; }

    [[nodiscard]] virtual uint32_t read_table_events(
        const resolved_event_query& /*query*/,
        uint32_t /*event_type*/,
        uint64_t /*after_id*/,
        uint64_t /*last_id*/,
        ph_table_event* /*rows*/,
        uint32_t /*capacity*/)
    {
        declined("read_table_events");
    }
    [[nodiscard]] virtual bool table_event_id_span(uint32_t /*event_type*/,
                                                   uint64_t& /*first_id*/,
                                                   uint64_t& /*last_id*/)
    {
        declined("table_event_id_span");
    }

    /** @brief The histogram of this file's tracks on the caller's grid. */
    [[nodiscard]] virtual reader_types::histogram_t histogram(
        const histogram_grid_t& grid) = 0;

protected:
    [[noreturn]] void declined(const char* call) const
    {
        throw std::logic_error{ m_format + ": " + call +
                                " is not available for this format" };
    }

private:
    std::string m_format;
};

}  // namespace profiler_hub
