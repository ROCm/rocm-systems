/**
 * @file profiler_hub.h
 * @brief Public C ABI for libprofiler-hub.
 *
 * A @ref ph_ctx_t is opaque and owns all storage returned through it (track
 * lists, node info, string fields, ...). Pointers obtained from a context
 * are only valid while that context is alive and must not be used after
 * ph_ctx_free() is called on it.
 *
 * This header is the reader API. The writer API is profiler-hub/cpp/writer.hpp.
 * The trimmer API is profiler_hub_trim.h.
 */

#ifndef PROFILER_HUB_INC
#define PROFILER_HUB_INC

#include "profiler_hub_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Detects whether @p file_path is a trace file and schema that
     *        profiler-hub supports, without loading the trace.
     * @param file_path Existing file to inspect.
     * @param probe Out parameter receiving the trace schema and schema version.
     *        Zeroed before probing.
     * @return PH_RESULT_SUCCESS when the file is a supported trace,
     *         PH_RESULT_UNSUPPORTED_FILE when it is not a supported format,
     *         PH_RESULT_INVALID_FILE when it is missing, unreadable, or not a
     *         usable trace, PH_RESULT_INVALID_ARGUMENT
     *         if an argument is null, PH_RESULT_INTERNAL_ERROR on an unexpected
     *         failure.
     * @note The file is opened read-only and is not created or modified.
     */
    ph_result_t ph_probe_file(const char* file_path, ph_file_probe_t* probe);

    /**
     * @brief Opens a trace file and creates a context for it.
     * @param ctx Out parameter receiving the new context. Must not be null.
     * @param file_path Path to the trace file to open. Must not be null.
     * @param future Null to read the trace before returning. Otherwise a future
     *        from ph_future_create() that was not passed to an operation yet: the
     *        call returns at once with the context and the trace is read in the
     *        background. The future reports the progress of the read in
     *        [0.0, 1.0], with a description of the current phase, and ends with
     *        PH_FUTURE_FINISHED, PH_FUTURE_CANCELLED or PH_FUTURE_ERROR.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p file_path is null
     *         or @p future already serves an operation,
     *         PH_RESULT_UNSUPPORTED_FILE or PH_RESULT_INVALID_FILE
     *         when the file is not a supported trace,
     *         PH_RESULT_CONTEXT_ALLOCATION_FAILED if the context could not be
     *         created or, without a future, the trace could not be read.
     * @note On success, the caller owns @p *ctx and must release it with
     *       ph_ctx_free(). On any failure @p *ctx is set to null.
     * @note While a context is being read, the calls that return its data wait
     *       for the read to end. If it was cancelled they return
     *       PH_RESULT_CANCELLED, if it failed PH_RESULT_INVALID_CONTEXT.
     * @warning ph_ctx_free() cancels a read that is still running and calls the
     *          on_finished callback of its future with PH_FUTURE_CANCELLED.
     */
    ph_result_t ph_ctx_create(ph_ctx_t* ctx, const char* file_path, ph_future_t future);

    /**
     * @brief One trace file and the GUID of the tables read from it. guid is
     *        empty for a format that has none.
     * @note path and guid point into the context that produced them. They
     *       stay valid until that context is freed.
     */
    typedef struct
    {
        const char* path;
        const char* guid;
    } ph_trace_source_t;

    /**
     * @brief Opens every trace file and creates one context for all of them.
     * @param paths Trace files of any supported format. A rocpd file with
     *        several metadata tables becomes one source per GUID. Must not be null.
     * @param path_count Number of entries in @p paths. Must be greater than zero.
     * @param future Same contract as ph_ctx_create().
     * @return Same status values as ph_ctx_create().
     */
    ph_result_t ph_ctx_create_files(ph_ctx_t*           ctx,
                                    const char* const* paths,
                                    uint32_t           path_count,
                                    ph_future_t        future);

    /**
     * @brief Opens every trace file and keeps timeline data in a cache under
     *        @p cache_directory. Null or empty leaves timeline data in memory,
     *        the same as ph_ctx_create_files().
     * @param cache_directory Application configuration folder. The cache is
     *        created at <cache_directory>/<folder>/timeline. The folder name
     *        is built by ph_trace_cache_directory().
     *        A later open of the same files reads the timeline from that
     *        cache. Table queries still read the trace database.
     */
    ph_result_t ph_ctx_create_files_cached(ph_ctx_t*           ctx,
                                           const char* const* paths,
                                           uint32_t           path_count,
                                           const char*        cache_directory,
                                           ph_future_t        future);

    /**
     * @brief Folder created by ph_ctx_create_files_cached() for these files.
     *        It is <cache_directory>/<folder>, and holds timeline/, catalog/, and summary.bin.
     *        The folder name is <date>_<format>_<version>_<name>_<size>: the
     *        newest last-write time to the second, the format and schema
     *        version from the first file, the first file's name, and the total
     *        size. The cache writer uses the same name, so @p paths must be the
     *        same files in the same order.
     * @param cache_root Application configuration folder. Null or empty
     *        yields an empty path.
     * @param paths Source files. Null only when @p path_count is 0.
     * @param path_count Number of entries in @p paths. Zero yields an empty
     *        path.
     * @param directory Out buffer. Null asks only for the required size.
     * @param capacity In: size of @p directory in bytes, ignored when
     *        @p directory is null. Out: bytes required to hold the path and
     *        its terminating NUL, or 0 when a source file cannot be read.
     * @return PH_RESULT_SUCCESS when @p directory receives the path, or when
     *         @p directory is null and @p capacity receives the required size.
     *         PH_RESULT_INVALID_ARGUMENT if @p capacity is null, a path entry
     *         is null, or @p directory is too small (the required size is
     *         still written to @p capacity). PH_RESULT_INVALID_FILE if a
     *         source is missing or not a regular file. PH_RESULT_INTERNAL_ERROR
     *         on an unexpected failure.
     */
    ph_result_t ph_trace_cache_directory(const char*        cache_root,
                                         const char* const* paths,
                                         uint32_t           path_count,
                                         char*              directory,
                                         uint32_t*          capacity);

    /**
     * @brief Number of file-and-GUID sources opened by the context.
     * @param count Out parameter. One for a single-GUID file.
     */
    ph_result_t ph_get_trace_source_count(ph_ctx_t ctx, uint32_t* count);

    /**
     * @brief Path and GUID of one source.
     * @param index Zero-based index, less than the count from
     *        ph_get_trace_source_count().
     */
    ph_result_t ph_get_trace_source(ph_ctx_t            ctx,
                                    uint32_t            index,
                                    ph_trace_source_t*  source);

    /**
     * @brief Releases a context and all data obtained through it.
     * @param ctx Context to release.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null.
     * @warning Any pointer previously returned through @p ctx (track lists,
     *          node info, string fields) is invalidated by this call.
     */
    ph_result_t ph_ctx_free(ph_ctx_t ctx);

    /**
     * @brief Reads the version of libprofiler-hub itself.
     * @param ctx Context to query.
     * @param version Out parameter receiving the library version.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p version is
     *         null.
     */
    ph_result_t ph_get_library_version(ph_ctx_t ctx, ph_library_version_t* version);

    /**
     * @brief Reads the schema version of the trace opened in @p ctx.
     * @param ctx Context to query.
     * @param version Out parameter receiving the schema version.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p version is
     *         null.
     */
    ph_result_t ph_get_schema_version(ph_ctx_t ctx, ph_schema_version_t* version);

    /**
     * @brief Retrieves the list of tracks contained in the trace.
     * @param ctx Context to query.
     * @param track_list Out parameter receiving the track list.
     * @param future Null to run the call before returning. Otherwise a future from
     *        ph_future_create() that was not passed to an operation yet: the call
     *        returns at once with PH_RESULT_SUCCESS, and the work runs in the
     *        background. The future ends with the ph_result_t the call would have
     *        returned (see ph_future_result()), or with PH_FUTURE_CANCELLED. The
     *        out parameter may be read only after the future ended with
     *        PH_FUTURE_FINISHED. A future that already serves an operation gives
     *        PH_RESULT_INVALID_ARGUMENT.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p track_list is
     *         null.
     * @note @p track_list->tracks and every ph_track_t::track_name in it
     *       point into memory owned by @p ctx. They remain valid until
     *       @p ctx is freed.
     */
    ph_result_t ph_get_track_list(ph_ctx_t         ctx,
                                  ph_track_list_t* track_list,
                                  ph_future_t      future);

    /**
     * @brief Retrieves node information (agents, processes, threads, streams,
     *        queues, PMC info, tracks, and stream-to-hardware links) for the
     *        trace.
     * @param ctx Context to query.
     * @param node Out parameter receiving the node info.
     * @param future Null to run the call before returning. Otherwise a future from
     *        ph_future_create() that was not passed to an operation yet: the call
     *        returns at once with PH_RESULT_SUCCESS, and the work runs in the
     *        background. The future ends with the ph_result_t the call would have
     *        returned (see ph_future_result()), or with PH_FUTURE_CANCELLED. The
     *        out parameter may be read only after the future ended with
     *        PH_FUTURE_FINISHED. A future that already serves an operation gives
     *        PH_RESULT_INVALID_ARGUMENT.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p node is null.
     * @note String and array fields of @p node point into memory owned by
     *       @p ctx and follow the same lifetime rule as ph_get_track_list().
     */
    ph_result_t ph_get_node(ph_ctx_t ctx, ph_node_t* node, ph_future_t future);

    /**
     * @brief Node information for one file and GUID.
     * @param index Source index from ph_get_trace_source().
     * @param node Out parameter. Its track list is the tracks whose
     *        ph_track_t::source_index equals @p index.
     * @param future Same contract as ph_get_node().
     */
    ph_result_t ph_get_source_node(ph_ctx_t    ctx,
                                   uint32_t    index,
                                   ph_node_t*  node,
                                   ph_future_t future);

    /**
     * @brief Retrieves duration events (region/kernel dispatch/memory
     *        copy/memory allocate) for a track within an optional time
     *        window.
     * @param ctx Context to query.
     * @param track_id Id of a track whose category is not
     *        PH_TRACK_CATEGORY_PMC_AGENT, as returned by
     *        ph_get_track_list()/ph_get_node(). A PMC track yields an empty
     *        list.
     * @param start_ts Start of the time window (ns). With @p end_ts also 0
     *        there is no filter.
     * @param end_ts End of the time window (ns). With only @p end_ts 0 the
     *        window has no upper bound. Events overlapping the window are
     *        returned.
     * @param events Out parameter receiving the event list; it is set to an
     *        empty list before any other work, so it is valid on failure.
     * @param future Null to run the call before returning. Otherwise a future from
     *        ph_future_create() that was not passed to an operation yet: the call
     *        returns at once with PH_RESULT_SUCCESS, and the work runs in the
     *        background. The future ends with the ph_result_t the call would have
     *        returned (see ph_future_result()), or with PH_FUTURE_CANCELLED. The
     *        out parameter may be read only after the future ended with
     *        PH_FUTURE_FINISHED. A future that already serves an operation gives
     *        PH_RESULT_INVALID_ARGUMENT.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p events is
     *         null or @p track_id does not identify a known track,
     *         PH_RESULT_INTERNAL_ERROR on failure.
     * @note @p events->events and every ph_event_t::name in it point into
     *       memory owned by @p ctx. A list read from the on-disk track cache
     *       stays valid until ph_release_track_events() or ph_ctx_free().
     *       Releasing it frees that window. Names stay valid until
     *       ph_ctx_free(). A list served from the in-memory track cache stays
     *       valid until ph_ctx_free(), and ph_release_track_events() only
     *       clears the caller's struct. A later call does not invalidate an
     *       earlier result that has not been released. The arrays are safe
     *       to read concurrently from multiple calls (including calls made
     *       through a future).
     * @note ph_event_t::track_id is @p track_id. A host thread omits rows
     *       that resolve to a device queue or stream. A stream or queue
     *       track keeps the rows its own query selected.
     * @note A request for a whole track (@p start_ts and @p end_ts both 0)
     *       from the in-memory track cache is answered from memory after the
     *       first call: repeated calls for the same track return the same
     *       storage, so the arrays must be treated as read-only. The on-disk
     *       track cache reads that range again and the caller releases it.
     *       Events of a whole track are ordered by ph_event_t::start. A
     *       request with a time window gets its own private storage; its
     *       events are the overlapping ones of the whole track, in the same
     *       order, with the same ph_event_t::nesting_depth.
     * @note An on-disk cache read checks @p future between checkpoints. A
     *       cancel during that read ends the future with PH_FUTURE_CANCELLED
     *       and leaves @p events empty.
     */
    ph_result_t ph_get_track_events(ph_ctx_t         ctx,
                                    uint32_t         track_id,
                                    uint64_t         start_ts,
                                    uint64_t         end_ts,
                                    ph_event_list_t* events,
                                    ph_future_t      future);

    /**
     * @brief Frees a track-event list read from the on-disk track cache.
     * @param events List returned by ph_get_track_events(). Cleared on
     *        success. May be empty.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p events is null.
     * @note A list from the in-memory track cache is not freed. The caller's
     *       struct is still cleared. Names are not freed.
     */
    ph_result_t ph_release_track_events(ph_ctx_t ctx, ph_event_list_t* events);

    /**
     * @brief Retrieves events correlated with @p event by matching stack id.
     *        The event itself is not included. A missing stack id, a stack id
     *        of 0, or an id that is not in the trace yields an empty list.
     *        A region reaches regions, kernel dispatches, memory copies, and
     *        memory allocations. A kernel dispatch reaches regions and kernel
     *        dispatches. A memory copy reaches regions and memory copies. A
     *        memory allocation reaches regions.
     * @param ctx Context to query.
     * @param event Event returned by ph_get_track_events(). @p event->id and
     *        @p event->type select the row. @p event->track_id selects the
     *        file when that track belongs to one of the open sources and the
     *        row is in that file. The object does not need to stay alive
     *        after this call returns.
     * @param events Out parameter receiving the correlated events; it is set to
     *        an empty list before any other work, so it is valid on failure.
     * @param future Null to run the call before returning. Otherwise a future from
     *        ph_future_create() that was not passed to an operation yet: the call
     *        returns at once with PH_RESULT_SUCCESS, and the work runs in the
     *        background. The future ends with the ph_result_t the call would have
     *        returned (see ph_future_result()), or with PH_FUTURE_CANCELLED. The
     *        out parameter may be read only after the future ended with
     *        PH_FUTURE_FINISHED. A future that already serves an operation gives
     *        PH_RESULT_INVALID_ARGUMENT.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p event or
     *         @p events is null or @p event->type is not a duration event,
     *         PH_RESULT_INTERNAL_ERROR on failure.
     * @note @p events->events and every ph_event_t::name in it point into
     *       memory owned by @p ctx and remain valid until @p ctx is freed.
     *       A later call does not invalidate an earlier result. Each call gets
     *       its own storage, safe to read concurrently.
     * @note ph_event_t::nesting_depth is the depth of that event on its own
     *       track, the same value ph_get_track_events() reports.
     *       ph_event_t::track_id is ph_track_t::id of that track. The track is
     *       loaded if it was not loaded already. An event that cannot be
     *       placed on a track is left out of the list.
     */
    ph_result_t ph_get_correlated_events(ph_ctx_t          ctx,
                                         const ph_event_t* event,
                                         ph_event_list_t*  events,
                                         ph_future_t       future);

    /**
     * @brief Reconstructs the overlapping-region call stack for @p event.
     *        A region event is the top frame, followed by the active caller
     *        regions on its track. The name of each region is the frame
     *        function. Optional source context in rocpd_event.line_info
     *        supplies file and line data. A non-region event or an id that is
     *        not in the trace yields an empty list.
     * @param ctx Context to query.
     * @param event Event returned by ph_get_track_events(). @p event->id and
     *        @p event->type select the row. @p event->track_id selects the
     *        file when that track belongs to one of the open sources and the
     *        row is in that file. The object does not need to stay alive
     *        after this call returns.
     * @param call_stack Out parameter receiving the frames; it is set to an
     *        empty list before any other work, so it is valid on failure.
     *        Index 0 is the top of the stack, the most recent call.
     * @param future Same contract as ph_get_correlated_events().
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p event or
     *         @p call_stack is null or @p event->type is not a duration event.
     * @note @p call_stack->frames and every string in those frames point into
     *       memory owned by @p ctx and remain valid until @p ctx is freed.
     *       A later call does not invalidate an earlier result.
     */
    ph_result_t ph_get_call_stack(ph_ctx_t          ctx,
                                  const ph_event_t* event,
                                  ph_call_stack_t*  call_stack,
                                  ph_future_t       future);

    /**
     * @brief Retrieves region_data_t for @p event.
     * @param event Event identified by id and type. On success, track_id is set
     *        to the track this row is placed on. The object must stay alive
     *        until the call, or the future, has finished.
     * @param details Out parameter. is_valid is 0 before the call returns and
     *        stays 0 when the event is not a region or the row is missing.
     * @param future Same contract as ph_get_correlated_events().
     * @return PH_RESULT_SUCCESS on success, including a missing row.
     *         PH_RESULT_INVALID_CONTEXT if @p ctx is null.
     *         PH_RESULT_INVALID_ARGUMENT if @p event or @p details is null or
     *         @p event->type is not PH_EVENT_TYPE_REGION.
     * @note String pointers and @p details->args point into memory owned by
     *       @p ctx and remain valid until @p ctx is freed. A later call does
     *       not invalidate an earlier result.
     */
    ph_result_t ph_get_region_details(ph_ctx_t             ctx,
                                      ph_event_t*          event,
                                      ph_region_details_t* details,
                                      ph_future_t          future);

    /**
     * @brief Retrieves kernel_dispatch_data_t for @p event.
     * @param details Out parameter. is_valid stays 0 when the event is not a
     *        kernel dispatch or the row is missing.
     * @param future Same contract as ph_get_correlated_events().
     * @return PH_RESULT_SUCCESS on success, including a missing row.
     *         PH_RESULT_INVALID_ARGUMENT if @p event->type is not
     *         PH_EVENT_TYPE_KERNEL_DISPATCH.
     * @note String pointers point into memory owned by @p ctx until
     *       ph_ctx_free(). A later call does not invalidate an earlier result.
     */
    ph_result_t ph_get_kernel_dispatch_details(ph_ctx_t                      ctx,
                                               ph_event_t*                   event,
                                               ph_kernel_dispatch_details_t* details,
                                               ph_future_t                   future);

    /**
     * @brief Retrieves memory_copy_data_t for @p event.
     * @param details Out parameter. is_valid stays 0 when the event is not a
     *        memory copy or the row is missing.
     * @param future Same contract as ph_get_correlated_events().
     * @return PH_RESULT_SUCCESS on success, including a missing row.
     *         PH_RESULT_INVALID_ARGUMENT if @p event->type is not
     *         PH_EVENT_TYPE_MEMORY_COPY.
     * @note String pointers point into memory owned by @p ctx until
     *       ph_ctx_free(). A later call does not invalidate an earlier result.
     */
    ph_result_t ph_get_memory_copy_details(ph_ctx_t                  ctx,
                                           ph_event_t*               event,
                                           ph_memory_copy_details_t* details,
                                           ph_future_t               future);

    /**
     * @brief Retrieves memory_alloc_data_t for @p event.
     * @param details Out parameter. is_valid stays 0 when the event is not a
     *        memory allocation or the row is missing.
     * @param future Same contract as ph_get_correlated_events().
     * @return PH_RESULT_SUCCESS on success, including a missing row.
     *         PH_RESULT_INVALID_ARGUMENT if @p event->type is not
     *         PH_EVENT_TYPE_MEMORY_ALLOCATE.
     * @note String pointers point into memory owned by @p ctx until
     *       ph_ctx_free(). A later call does not invalidate an earlier result.
     */
    ph_result_t ph_get_memory_alloc_details(ph_ctx_t                   ctx,
                                            ph_event_t*                event,
                                            ph_memory_alloc_details_t* details,
                                            ph_future_t                future);

    /**
     * @brief Retrieves PMC/counter samples for a track within an optional
     *        time window.
     * @param ctx Context to query.
     * @param track_id Id of a track whose category is
     *        PH_TRACK_CATEGORY_PMC_AGENT, as returned by
     *        ph_get_track_list()/ph_get_node(). Any other track yields an
     *        empty list.
     * @param start_ts Start of the time window (ns). With @p end_ts also 0
     *        there is no filter.
     * @param end_ts End of the time window (ns). With only @p end_ts 0 the
     *        window has no upper bound.
     * @param samples Out parameter receiving the sample list; it is set to
     *        an empty list before any other work, so it is valid on failure.
     * @param future Null to run the call before returning. Otherwise a future from
     *        ph_future_create() that was not passed to an operation yet: the call
     *        returns at once with PH_RESULT_SUCCESS, and the work runs in the
     *        background. The future ends with the ph_result_t the call would have
     *        returned (see ph_future_result()), or with PH_FUTURE_CANCELLED. The
     *        out parameter may be read only after the future ended with
     *        PH_FUTURE_FINISHED. A future that already serves an operation gives
     *        PH_RESULT_INVALID_ARGUMENT.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p samples is
     *         null or @p track_id does not identify a known track,
     *         PH_RESULT_INTERNAL_ERROR on failure.
     * @note @p samples->samples points into memory owned by @p ctx. A list
     *       read from the on-disk track cache stays valid until
     *       ph_release_track_samples() or ph_ctx_free(). A list served from
     *       the in-memory track cache stays valid until ph_ctx_free(). A
     *       later call does not invalidate an earlier result that has not
     *       been released. Each call gets its own private storage, safe to
     *       read concurrently from multiple calls (including calls made
     *       through a future).
     * @note A request for a whole track (@p start_ts and @p end_ts both 0)
     *       from the in-memory track cache is answered from memory after the
     *       first call: repeated calls for the same track return the same
     *       storage, so the array must be treated as read-only. The on-disk
     *       track cache reads that range again and the caller releases it.
     *       Samples of a whole track are ordered by ph_sample_t::timestamp.
     *       A request with a time window gets its own private storage and
     *       includes the nearest sample before the window and the nearest
     *       sample after it. When no sample follows the window, the last
     *       value is repeated at the track end.
     */
    ph_result_t ph_get_track_samples(ph_ctx_t          ctx,
                                     uint32_t          track_id,
                                     uint64_t          start_ts,
                                     uint64_t          end_ts,
                                     ph_sample_list_t* samples,
                                     ph_future_t       future);

    /**
     * @brief Frees a sample list read from the on-disk track cache.
     * @param samples List returned by ph_get_track_samples(). Cleared on
     *        success. May be empty.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p samples is null.
     * @note A list from the in-memory track cache is not freed. The caller's
     *       struct is still cleared.
     */
    ph_result_t ph_release_track_samples(ph_ctx_t ctx, ph_sample_list_t* samples);

    /**
     * @brief Builds one histogram per track on a shared time grid.
     * @param ctx Context to query.
     * @param max_buckets Upper bound on the number of buckets. Must be non-zero.
     *        The grid never has more buckets than this. A short trace has fewer.
     * @param histogram Out parameter. It is cleared before any other work.
     * @param future Null to run the call before returning. Otherwise a future
     *        from ph_future_create() that was not passed to an operation yet.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p histogram is
     *         null or @p max_buckets is 0, PH_RESULT_INTERNAL_ERROR on failure.
     * @note Returned arrays are owned by @p ctx, remain valid until @p ctx is
     *       freed, and are not invalidated by a later call.
     */
    ph_result_t ph_get_histogram(ph_ctx_t        ctx,
                                 uint32_t        max_buckets,
                                 ph_histogram_t* histogram,
                                 ph_future_t     future);

    /**
     * @brief Creates a future that can be passed to an API call to run it
     *        asynchronously.
     * @param on_progress Called with the fraction completed and a description of
     *        the current work. The description may be null. The callback itself
     *        may be null.
     * @param on_finished Called once when the operation ends, may be null.
     * @param future Out parameter receiving the new future. Must not be null.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_ARGUMENT if
     *         @p future is null, PH_RESULT_FUTURE_ALLOCATION_FAILED if it could
     *         not be allocated.
     * @note The caller owns @p *future and must release it with
     *       ph_future_free(). A future serves exactly one operation.
     * @warning The callbacks run on a worker thread. They must not call
     *          ph_future_wait() on their own future or free the context.
     */
    ph_result_t ph_future_create(ph_progress_fn on_progress,
                                 ph_finished_fn on_finished,
                                 ph_future_t*   future);

    /**
     * @brief Blocks until the operation of @p future has ended and its
     *        on_finished callback has returned.
     * @param future Future to wait on.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_ARGUMENT if
     *         @p future is null or was not passed to an operation yet.
     */
    ph_result_t ph_future_wait(ph_future_t future);

    /**
     * @brief Requests cancellation of @p future.
     * @param future Future to cancel.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_ARGUMENT if
     *         @p future is null.
     * @note Cancellation is cooperative and does not require the future to be
     *       attached. A synchronous call such as ph_event_query_read observes
     *       the stop without attaching the future, so the same future can be
     *       reused. The SQLite statement that call is running is interrupted.
     *       An asynchronous operation that has not started is skipped. A
     *       running one stops at its next check, or when its statement is
     *       interrupted, and ends with PH_FUTURE_CANCELLED. Cancelling an
     *       operation that already ended does not change its result.
     */
    ph_result_t ph_future_cancel(ph_future_t future);

    /**
     * @brief Reads the ph_result_t the operation of @p future ended with.
     * @param future Future to query.
     * @param result Out parameter receiving the result.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_ARGUMENT if
     *         @p future or @p result is null, or the operation has not ended.
     */
    ph_result_t ph_future_result(ph_future_t future, ph_result_t* result);

    /**
     * @brief Releases @p future.
     * @param future Future to release.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_ARGUMENT if
     *         @p future is null or was already released.
     * @note Does not wait for the operation. A running operation keeps the
     *       future alive until its on_finished callback has returned.
     */
    ph_result_t ph_future_free(ph_future_t future);

    /**
     * @brief Resolves @p filter against the open trace.
     *        Search strings are matched in the loaded string table and kernel
     *        symbol names. Later table reads use those ids.
     * @param ctx Context whose catalog supplies the string table.
     * @param filter Filter filled by the caller. filter->size must be
     *        sizeof(ph_event_filter_t).
     * @param query Out parameter. Set to null on failure. The caller releases
     *        a successful query with ph_event_query_free().
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null or the trace is not loaded, PH_RESULT_INVALID_ARGUMENT
     *         if @p filter or @p query is null, @p filter->size is too small, or
     *         a list pointer is null while its count is non-zero,
     *         PH_RESULT_CANCELLED if loading the trace was cancelled.
     * @note Does not read event rows. A string filter that matches nothing is
     *       still success; the query then selects no rows.
     */
    ph_result_t ph_event_query_create(ph_ctx_t                 ctx,
                                      const ph_event_filter_t* filter,
                                      ph_event_query_t*        query);

    /**
     * @brief Releases a query from ph_event_query_create().
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_ARGUMENT if
     *         @p query is null.
     */
    ph_result_t ph_event_query_free(ph_event_query_t query);

    /**
     * @brief Reads the next page of table rows for one event type.
     * @param query Query from ph_event_query_create().
     * @param event_type ph_event_type_t value. A type the query does not list
     *        yields a successful empty page.
     * @param after_id Return rows whose id is greater than this. Pass 0 to
     *        start. Pass the last returned id to continue.
     * @param last_id Return rows whose id is not greater than this. Pass
     *        UINT64_MAX for no upper bound. Ranges from
     *        ph_event_query_id_span() let several threads read one table.
     * @param rows Caller-owned buffer. Filled with complete values; nothing
     *        in a row points at profiler-hub memory.
     * @param capacity Number of entries @p rows can hold. Must be non-zero.
     * @param count Out parameter. Set to the number of rows written. A count
     *        smaller than @p capacity means there are no further rows.
     * @param future Optional cancellation source. This synchronous call does
     *        not attach or complete it, so the same future can be reused.
     *        ph_future_cancel requests the stop this call observes and
     *        interrupts the SQLite statement reading the page.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_ARGUMENT if a
     *         pointer is null, @p capacity is 0, or @p event_type is unknown,
     *         PH_RESULT_INVALID_CONTEXT if the query has no trace,
     *         PH_RESULT_CANCELLED if @p future was cancelled.
     */
    ph_result_t ph_event_query_read(ph_event_query_t  query,
                                    uint32_t          event_type,
                                    uint64_t          after_id,
                                    uint64_t          last_id,
                                    ph_table_event_t* rows,
                                    uint32_t          capacity,
                                    uint32_t*         count,
                                    ph_future_t       future);

    /**
     * @brief Reads the smallest and largest row id of the table that holds
     *        @p event_type, ignoring the query filter.
     * @param first_id Out parameter. Set to 0 if the table is empty.
     * @param last_id Out parameter. Set to 0 if the table is empty.
     * @param future Optional cancellation source. This synchronous call does
     *        not attach or complete it. ph_future_cancel requests the stop
     *        this call observes and interrupts its SQLite statement.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_ARGUMENT if a
     *         pointer is null or @p event_type is unknown,
     *         PH_RESULT_INVALID_CONTEXT if the query has no trace,
     *         PH_RESULT_CANCELLED if @p future was cancelled.
     */
    ph_result_t ph_event_query_id_span(ph_event_query_t query,
                                       uint32_t         event_type,
                                       uint64_t*        first_id,
                                       uint64_t*        last_id,
                                       ph_future_t      future);

    /** @brief One display name and the duration aggregates of its events.
     *         @p name is valid only for the duration of the callback. */
    typedef struct ph_name_duration_summary
    {
        const char* name;
        uint64_t    count;
        uint64_t    min_duration;
        uint64_t    max_duration;
        uint64_t    total_duration;
    } ph_name_duration_summary_t;

    typedef void (*ph_name_duration_summary_callback_t)(
        const ph_name_duration_summary_t* row,
        void*                             user_data);

    /**
     * @brief Aggregates matching events by display name inside SQLite.
     *        Durations are end minus start, in trace ticks.
     * @param future Optional cancellation source. This synchronous call does
     *        not attach or complete it. ph_future_cancel requests the stop
     *        this call observes and interrupts its SQLite statement.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_ARGUMENT if a
     *         pointer is null or @p event_type is unknown,
     *         PH_RESULT_INVALID_CONTEXT if the query has no trace,
     *         PH_RESULT_UNSUPPORTED_FILE if this trace format has no summary
     *         query, PH_RESULT_CANCELLED if @p future was cancelled.
     */
    ph_result_t ph_event_query_summarize_durations(
        ph_event_query_t                      query,
        uint32_t                              event_type,
        ph_name_duration_summary_callback_t   callback,
        void*                                 user_data,
        ph_future_t                           future);

    /**
     * @brief Looks up text the trace catalog already loaded.
     * @param kind PH_TEXT_KIND_STRING or PH_TEXT_KIND_KERNEL_SYMBOL.
     * @param text Out parameter. Points at catalog memory owned by @p ctx.
     *        Valid until @p ctx is freed.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_ARGUMENT if a
     *         pointer is null, @p kind is unknown, or @p id is not in the
     *         catalog, PH_RESULT_INVALID_CONTEXT if @p ctx is null or the
     *         trace is not loaded.
     */
    ph_result_t ph_lookup_text(ph_ctx_t     ctx,
                               uint64_t     id,
                               uint32_t     kind,
                               const char** text);

    /**
     * @brief Receives one catalog entry from ph_visit_text.
     * @param text Catalog memory owned by the context. Valid until that
     *        context is freed.
     * @param user_data The pointer passed to ph_visit_text.
     */
    typedef void (*ph_text_visitor_t)(uint64_t id, const char* text, void* user_data);

    /**
     * @brief Calls @p visitor once for each catalog entry of @p kind.
     * @param kind PH_TEXT_KIND_STRING or PH_TEXT_KIND_KERNEL_SYMBOL.
     * @param user_data Forwarded to every call. May be null.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_ARGUMENT if
     *         @p visitor is null or @p kind is unknown,
     *         PH_RESULT_INVALID_CONTEXT if @p ctx is null or the trace is
     *         not loaded.
     * @note @p visitor must not free @p ctx or start another operation on it.
     */
    ph_result_t ph_visit_text(ph_ctx_t           ctx,
                              uint32_t           kind,
                              ph_text_visitor_t  visitor,
                              void*              user_data);

    /**
     * @brief Same as ph_visit_text() for one file and GUID.
     * @param source Source index from ph_get_trace_source().
     */
    ph_result_t ph_visit_source_text(ph_ctx_t          ctx,
                                     uint32_t          source,
                                     uint32_t          kind,
                                     ph_text_visitor_t visitor,
                                     void*             user_data);

#ifdef __cplusplus
}
#endif

#endif  // #ifndef PROFILER_HUB_INC
