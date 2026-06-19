// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <profiler-hub/reader_types.hpp>
#include <profiler-hub/storage.hpp>

#include <memory>
#include <optional>
#include <string>

namespace profiler_hub
{

// ============================================================================
// Reader Interface
// ============================================================================

struct reader_t
{
    /**
     * @brief Construct a reader with the given storage backend
     * @param storage Storage backend to read from (takes ownership)
     */
    explicit reader_t(std::unique_ptr<profiler_hub::storage_t> storage);

    ~reader_t();

    reader_t()                           = delete;
    reader_t(const reader_t&)            = delete;
    reader_t& operator=(const reader_t&) = delete;
    reader_t(reader_t&&)                 = delete;
    reader_t& operator=(reader_t&&)      = delete;

    /**
     *@section Info Table Accessors (Eagerly Loaded, Cached)
     * These are loaded at construction and cached for the session lifetime.
     * Returns shared_ptr to cached objects; fast access, no database query.
     */

    /**
     * @brief Get all node info from cache
     * @return List of all node info objects
     */
    [[nodiscard]] reader_types::node_info_list_t get_all_nodes() const;

    /**
     * @brief Get all process info from cache
     * @return List of all process info objects
     */
    [[nodiscard]] reader_types::process_info_list_t get_all_processes() const;

    /**
     * @brief Get all thread info from cache
     * @return List of all thread info objects
     */
    [[nodiscard]] reader_types::thread_info_list_t get_all_threads() const;

    /**
     * @brief Get all agent info from cache
     * @return List of all agent info objects
     */
    [[nodiscard]] reader_types::agent_info_list_t get_all_agents() const;

    /**
     * @brief Get all queue info from cache
     * @return List of all queue info objects
     */
    [[nodiscard]] reader_types::queue_info_list_t get_all_queues() const;

    /**
     * @brief Get all stream info from cache
     * @return List of all stream info objects
     */
    [[nodiscard]] reader_types::stream_info_list_t get_all_streams() const;

    /**
     * @brief Get all PMC info from cache
     * @return List of all PMC info objects
     */
    [[nodiscard]] reader_types::pmc_info_list_t get_all_pmc_info() const;

    /**
     * @brief Get all code object info from cache
     * @return List of all code object info objects
     */
    [[nodiscard]] reader_types::code_object_info_list_t get_all_code_objects() const;

    /**
     * @brief Get all kernel symbol info from cache
     * @return List of all kernel symbol info objects
     */
    [[nodiscard]] reader_types::kernel_symbol_info_list_t get_all_kernel_symbols() const;

    /**
     *@section Track Accessors (Eagerly Loaded, Cached)
     * Tracks organize events on the timeline. Each track represents a unique
     * context (e.g., node+process+thread, or node+agent+queue).
     */

    /**
     * @brief Get all track info from cache
     * @return List of all track info objects
     */
    [[nodiscard]] reader_types::track_info_list_t get_all_tracks() const;

    /**
     *@section Timeline Event Queries (On-Demand, Not Stored)
     * These query the database and return lightweight timeline_event_t for display.
     * The returned vector is owned by the caller; reader does not cache events.
     * Use get_event_details() to fetch full data for a specific event.
     */

    /**
     * @brief Get all events for a track within optional time window
     * @param track Track to query events for
     * @param filter Optional filter for time window and pagination
     * @return List of lightweight timeline events for display
     */
    [[nodiscard]] reader_types::timeline_event_list_t get_events_for_track(
        reader_types::track_info_ptr_t      track,
        const reader_types::event_filter_t& filter = {}) const;

    /**
     * @brief Get events across all tracks matching filter
     * @param filter Optional filter for time window and pagination
     * @return List of lightweight timeline events for display
     * @note For large databases, use pagination to limit result size
     */
    [[nodiscard]] reader_types::timeline_event_list_t get_events(
        const reader_types::event_filter_t& filter = {}) const;

    /**
     * @brief Get total count of events matching filter without fetching data
     * @param filter Optional filter; honors `time_window` and `types`. Pagination,
     *               sort, and where fields are ignored - count reflects the total
     *               number of matches.
     * @return Number of matching events
     * @note Useful for pagination UI - returns the unpaginated total.
     */
    [[nodiscard]] size_t get_event_count(
        const reader_types::event_filter_t& filter = {}) const;

    /**
     *@section Event Details (On-Demand Query by db_id)
     * Fetch full details for a timeline event. Queries database by db_id.
     * Returns nullopt if type mismatch or db_id not found.
     */

    /**
     * @brief Get full region details for a timeline event
     * @param event Timeline event to fetch details for
     * @return Region data, or nullopt if type mismatch or not found
     */
    [[nodiscard]] std::optional<reader_types::region_data_t> get_region_details(
        const reader_types::timeline_event_t& event) const;

    /**
     * @brief Get full kernel dispatch details for a timeline event
     * @param event Timeline event to fetch details for
     * @return Kernel dispatch data, or nullopt if type mismatch or not found
     */
    [[nodiscard]] std::optional<reader_types::kernel_dispatch_data_t>
    get_kernel_dispatch_details(const reader_types::timeline_event_t& event) const;

    /**
     * @brief Get full memory copy details for a timeline event
     * @param event Timeline event to fetch details for
     * @return Memory copy data, or nullopt if type mismatch or not found
     */
    [[nodiscard]] std::optional<reader_types::memory_copy_data_t> get_memory_copy_details(
        const reader_types::timeline_event_t& event) const;

    /**
     * @brief Get full memory alloc details for a timeline event
     * @param event Timeline event to fetch details for
     * @return Memory alloc data, or nullopt if type mismatch or not found
     */
    [[nodiscard]] std::optional<reader_types::memory_alloc_data_t>
    get_memory_alloc_details(const reader_types::timeline_event_t& event) const;

    /**
     * @brief Get full sample details for a timeline event
     * @param event Timeline event to fetch details for
     * @return Sample data, or nullopt if type mismatch or not found
     */
    [[nodiscard]] std::optional<reader_types::sample_data_t> get_sample_details(
        const reader_types::timeline_event_t& event) const;

    /**
     * @brief Get full PMC event details for a timeline event
     * @param event Timeline event to fetch details for
     * @return PMC event data, or nullopt if type mismatch or not found
     */
    [[nodiscard]] std::optional<reader_types::pmc_event_data_t> get_pmc_event_details(
        const reader_types::timeline_event_t& event) const;

    /**
     *@section Event Property Accessors (On-Demand, Related Data)
     * Fetch additional properties for a specific event.
     * These perform database queries on demand.
     */

    /**
     * @brief Get call stack for an event
     * @param event Timeline event to fetch call stack for
     * @return Call stack data (empty if not available in database)
     */
    [[nodiscard]] reader_types::call_stack_t get_call_stack(
        const reader_types::timeline_event_t& event) const;

    /**
     * @brief Get source code context for an event
     * @param event Timeline event to fetch source context for
     * @return List of source context entries (empty if not available)
     */
    [[nodiscard]] reader_types::source_context_list_t get_source_context(
        const reader_types::timeline_event_t& event) const;

    /**
     * @brief Get function arguments for an event
     * @param event Timeline event to fetch arguments for (typically region events)
     * @return List of argument data (empty if not available)
     */
    [[nodiscard]] reader_types::arg_data_list_t get_arguments(
        const reader_types::timeline_event_t& event) const;

    /**
     * @brief Get correlated events via stack_id matching
     * @param event Timeline event to find correlations for
     * @return List of related events (e.g., CPU region -> GPU kernel correlation)
     * @note Finds events where stack_id matches and id differs (excludes self)
     */
    [[nodiscard]] reader_types::timeline_event_list_t get_correlated_events(
        const reader_types::timeline_event_t& event) const;

    /**
     *@section Summary/Statistics (Aggregate Queries)
     * Get aggregated statistics for events. These perform GROUP BY queries.
     */
    /**
     * @brief Get aggregated kernel dispatch statistics
     * @param window Optional time window to filter events
     * @return List of kernel summary statistics
     */
    [[nodiscard]] reader_types::event_summary_list_t get_kernel_summary(
        const reader_types::time_window_t& window = {}) const;

    /**
     * @brief Get aggregated region statistics
     * @param window Optional time window to filter events
     * @return List of region summary statistics
     */
    [[nodiscard]] reader_types::event_summary_list_t get_region_summary(
        const reader_types::time_window_t& window = {}) const;

    /**
     *@section Database Metadata
     * Get metadata about the database.
     */
    /**
     * @brief Get time range of all data in the database
     * @return Time window spanning all events
     */
    [[nodiscard]] reader_types::time_window_t get_data_time_range() const;

    /**
     * @brief Get total counts of each event type
     * @param window Optional time window to filter events
     * @return Counts for each event type
     */
    [[nodiscard]] reader_types::event_counts_t get_event_counts(
        const reader_types::time_window_t& window = {}) const;

private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace profiler_hub
