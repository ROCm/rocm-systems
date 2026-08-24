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
    [[nodiscard]] reader_types::track_info_list_t get_tracks() const;

    /**
     *@section Timeline Event Queries (On-Demand, Not Stored)
     * These query the database and return lightweight timeline_event_t for display.
     * The returned vector is owned by the caller; reader does not cache events.
     * Use get_event_info() to fetch full data for a specific event.
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
     *@section Track-Scoped Event Queries (On-Demand, Not Stored)
     * Fetch events scoped to a single track by its track_info_t::id. Unlike
     * get_events_for_track(), gpu_queue / dma / stream / memory / kernel_dispatch_pmc
     * tracks return ONLY the events for that specific queue, agent, or stream, not all
     * events sharing a (nid,pid,tid) context.
     */

    /**
     * @brief Get all interval events on a track, scoped to that track's identity
     * @param track_id Track identifier from track_info_t::id. Valid interval
     *                 track types: cpu_thread, gpu_queue, dma, stream, memory,
     *                 kernel_dispatch_pmc. counter and memory_activity are
     *                 scalar-only; they return empty — use get_scalar_track().
     * @param filter Optional time-window / pagination filter
     * @return Interval events ordered by start ascending, with lane (always
     *         valid packing row), level, and parent_id (containment; opaque
     *         handle, populated only on stack tracks) precomputed and per-event
     *         category resolved. Empty (not an error) if track_id is unknown or
     *         scalar-only.
     */
    [[nodiscard]] reader_types::interval_entry_list_t get_interval_track(
        reader_types::track_id_t            track_id,
        const reader_types::event_filter_t& filter = {}) const;

    /**
     * @brief Get all scalar (timestamp,value) samples on a counter or
     *        memory_activity track
     * @param track_id Track identifier from track_info_t::id. Valid scalar
     *                 track types: counter, memory_activity. All other track
     *                 types return empty — use get_interval_track() instead.
     * @param filter Optional time-window / pagination filter (types is ignored)
     * @return Scalar events ordered by timestamp ascending. Empty (not an
     *         error) if track_id is unknown or interval-only.
     */
    [[nodiscard]] reader_types::scalar_sample_list_t get_scalar_track(
        reader_types::track_id_t            track_id,
        const reader_types::event_filter_t& filter = {}) const;

    /**
     * @brief Get bounds/count for a track without loading its events
     * @param track_id Track identifier from track_info_t::id (any track type)
     * @return {min_ts, max_ts, count} computed via cheap SQL aggregates over the same
     *         events get_interval_track / get_scalar_track would return for this track.
     *         min_ts/max_ts are nullopt and count is 0 when track_id is unknown or the
     *         track has no events.
     * @note Cheap enough to call per-track at discovery; does not materialize event rows.
     */
    [[nodiscard]] reader_types::track_stats_t get_track_stats(
        reader_types::track_id_t track_id) const;

    /**
     * @brief Get all causal links between events across all tracks (post-hoc pass)
     * @param filter Optional time-window filter applied to the SOURCE event's start.
     *               pagination/sort/types are ignored.
     * @return Every (source -> dest) pair derivable from a shared non-zero stack_id,
     *         forming the full stack-clique. Each endpoint is an opaque event handle
     *         that internally encodes its event type, so no companion type tag is needed
     *         to disambiguate colliding per-type-table row ids. Emitted edge categories:
     *         region->region,
     *         region->{kernel_dispatch, memory_copy, memory_allocate}, and same-type
     *         siblings (kernel_dispatch->kernel_dispatch, memory_copy->memory_copy,
     *         memory_allocate->memory_allocate). May be one-to-many.
     */
    [[nodiscard]] reader_types::flow_list_t get_flows(
        const reader_types::event_filter_t& filter = {}) const;

    /**
     * @brief Get the directed flow edges adjacent to a single event.
     * @param id Event handle to match against either endpoint of each edge.
     * @return Every edge whose source or dest equals @p id. Post-filter over
     *         get_flows({}); cheap, no additional query.
     */
    [[nodiscard]] reader_types::flow_list_t get_flows_for_event(
        const reader_types::event_id_t& id) const;

    /**
     * @brief Get every edge in one causal chain (flow_id group).
     * @param flow_id Chain handle grouping edges that share a source stack_id.
     * @return Every edge whose flow_id equals @p flow_id. Sorting the result by
     *         source start recovers the chain's linear order. Post-filter over
     *         get_flows({}); cheap, no additional query.
     */
    [[nodiscard]] reader_types::flow_list_t get_flows_for_chain(
        const reader_types::flow_id_t& flow_id) const;

    /**
     * @brief Get the flow edges visible in a viewport, capped for dense views.
     * @param tracks Track ids in view. An edge is kept iff AT LEAST ONE endpoint sits on
     * a listed track (so a cross-track arrow with one endpoint just off-screen still
     * surfaces its visible half). An empty vector applies no track filter (all tracks).
     * @param window Time range. An edge is kept iff its temporal extent
     *               [min(src.start,dst.start), max(src.end,dst.end)] intersects
     *               [window.start or 0, window.end or +inf]. An empty window (both fields
     *               nullopt) applies no time filter.
     * @param max_edges Cap on the result. When >0 and the in-window/in-track set exceeds
     *               it, edges are decimated to the @p max_edges highest
     * arrow-span-latency (dst.start - src.end, clamped at 0) edges, tie-broken by
     * (source, dest) handle order so the kept set is STABLE across pans. 0 = no cap.
     * @return The kept edges. A returned edge still carries NO timestamps — endpoint
     *         geometry is used internally only to filter and rank; the emitted
     * flow_edge_t shape (source, dest, flow_id, kind) is unchanged. Every returned edge
     * is a member of get_flows({}).
     */
    [[nodiscard]] reader_types::flow_list_t get_flows_in_window(
        const std::vector<reader_types::track_id_t>& tracks,
        const reader_types::time_window_t&           window,
        uint32_t                                     max_edges) const;

    /**
     *@section Event Detail (On-Demand, Unified)
     * One collapsed detail path for every event type.
     */

    /**
     * @brief Get unified detail for any event, by its opaque handle.
     *
     * Dispatches on the handle's internal event type across all six event_type_t cases
     * (region, kernel_dispatch, memory_copy, memory_allocate, sample, pmc_event) and
     * returns a fixed common header (name, category, ts, te) plus a generic `properties`
     * bag of named, typed values (see event_info_t / arg_t). Point events (sample,
     * pmc_event) carry `te == std::nullopt`. Linked entities are emitted as integer-id
     * properties, not resolved sub-structs. Missing optional fields are omitted.
     *
     * @param id Opaque event handle (from get_interval_track / get_scalar_track / flows).
     * @return Unified detail, or nullopt if the handle names no known event.
     */
    [[nodiscard]] std::optional<reader_types::event_info_t> get_event_info(
        const reader_types::event_id_t& id) const;

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
     * @brief Get call stack for an event, by its opaque handle.
     *
     * Delegates to the timeline_event_t overload; event_id_t opacity is
     * preserved (no public type/row_id accessor is exposed).
     * @param id Opaque event handle.
     * @return Call stack data (empty if not available in database).
     */
    [[nodiscard]] reader_types::call_stack_t get_call_stack(
        const reader_types::event_id_t& id) const;

    /**
     * @brief Get source code context for an event, by its opaque handle.
     *
     * Opaque-handle overload of get_source_context; see get_call_stack(event_id_t)
     * for the opacity-preserving delegation rationale.
     * @param id Opaque event handle.
     * @return List of source context entries (empty if not available).
     */
    [[nodiscard]] reader_types::source_context_list_t get_source_context(
        const reader_types::event_id_t& id) const;

    /**
     * @brief Get function arguments for an event
     * @param event Timeline event to fetch arguments for (typically region events)
     * @return List of argument data (empty if not available)
     */
    [[nodiscard]] reader_types::arg_data_list_t get_arguments(
        const reader_types::timeline_event_t& event) const;

    /**
     * @brief Get function arguments for an event, by its opaque handle.
     *
     * Opaque-handle overload of get_arguments; see get_call_stack(event_id_t)
     * for the opacity-preserving delegation rationale. Unlike the folded
     * name/value pairs in event_info_t::properties, this preserves each
     * argument's position and type (arg_data_t).
     * @param id Opaque event handle.
     * @return List of argument data (empty if not available).
     */
    [[nodiscard]] reader_types::arg_data_list_t get_arguments(
        const reader_types::event_id_t& id) const;

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
    [[nodiscard]] reader_types::time_window_t get_time_range() const;

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
