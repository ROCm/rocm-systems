/**
 * @file profiler_hub.h
 * @brief Public C ABI for libprofiler-hub.
 *
 * A @ref ph_ctx_t is opaque and owns all storage returned through it (track
 * lists, node info, string fields, ...). Pointers obtained from a context
 * are only valid while that context is alive and must not be used after
 * ph_ctx_free() is called on it.
 */

#ifndef PROFILER_HUB_INC
#define PROFILER_HUB_INC

#include "profiler_hub_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Opens a trace file and creates a context for it.
     * @param ctx Out parameter receiving the new context. Must not be null.
     * @param file_path Path to the trace database to open. Must not be null.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p file_path is
     *         null, PH_RESULT_CONTEXT_ALLOCATION_FAILED if the trace could
     *         not be opened/parsed.
     * @note On success, the caller owns @p *ctx and must release it with
     *       ph_ctx_free(). On any failure @p *ctx is set to null.
     */
    ph_result_t ph_ctx_create(ph_ctx_t* ctx, const char* file_path);

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
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p track_list is
     *         null.
     * @note @p track_list->tracks and every ph_track_t::track_name in it
     *       point into memory owned by @p ctx. They remain valid until
     *       @p ctx is freed.
     */
    ph_result_t ph_get_track_list(ph_ctx_t ctx, ph_track_list_t* track_list);

    /**
     * @brief Retrieves node information (agents and tracks) for the trace.
     * @param ctx Context to query.
     * @param node Out parameter receiving the node info.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p node is null.
     * @note String and array fields of @p node point into memory owned by
     *       @p ctx and follow the same lifetime rule as ph_get_track_list().
     */
    ph_result_t ph_get_node(ph_ctx_t ctx, ph_node_t* node);

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
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p events is
     *         null or @p track_id does not identify a known track,
     *         PH_RESULT_INTERNAL_ERROR on failure.
     * @note @p events->events and every ph_event_t::name in it point into
     *       memory owned by @p ctx and remain valid until @p ctx is freed.
     *       Unlike ph_get_track_list()/ph_get_node(), a later call to this
     *       function does NOT invalidate an earlier one's result, and the
     *       arrays are safe to read concurrently from multiple calls
     *       (including calls made through a future).
     * @note A request for a whole track (@p start_ts and @p end_ts both 0)
     *       is answered from memory after the first call: repeated calls for
     *       the same track return the same storage, so the arrays must be
     *       treated as read-only. Events of a whole track are ordered by
     *       ph_event_t::start. A request with a time window gets its own
     *       private storage; its events are the overlapping ones of the whole
     *       track, in the same order, with the same ph_event_t::nesting_depth.
     */
    ph_result_t ph_get_track_events(ph_ctx_t         ctx,
                                    uint32_t         track_id,
                                    uint64_t         start_ts,
                                    uint64_t         end_ts,
                                    ph_event_list_t* events);

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
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p samples is
     *         null or @p track_id does not identify a known track,
     *         PH_RESULT_INTERNAL_ERROR on failure.
     * @note @p samples->samples points into memory owned by @p ctx and
     *       remains valid until @p ctx is freed. Unlike
     *       ph_get_track_list()/ph_get_node(), a later call to this
     *       function does NOT invalidate an earlier one's result -- each
     *       call gets its own private storage, safe to read concurrently
     *       from multiple calls (including calls made through a future).
     * @note A request for a whole track (@p start_ts and @p end_ts both 0)
     *       is answered from memory after the first call: repeated calls for
     *       the same track return the same storage, so the array must be
     *       treated as read-only. Samples of a whole track are ordered by
     *       ph_sample_t::timestamp. A request with a time window gets its own
     *       private storage.
     */
    ph_result_t ph_get_track_samples(ph_ctx_t          ctx,
                                     uint32_t          track_id,
                                     uint64_t          start_ts,
                                     uint64_t          end_ts,
                                     ph_sample_list_t* samples);

    /**
     * @brief Creates a future that can be passed to an API call to run it
     *        asynchronously.
     * @param on_progress Called with the progress of the operation, may be null.
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
     * @brief Requests cancellation of the operation of @p future.
     * @param future Future to cancel.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_ARGUMENT if
     *         @p future is null or was not passed to an operation yet.
     * @note Cancellation is cooperative: an operation that has not started is
     *       skipped, a running one stops at its next check. The operation ends
     *       with PH_FUTURE_CANCELLED. Cancelling an operation that already
     *       ended has no effect.
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

#ifdef __cplusplus
}
#endif

#endif  // #ifndef PROFILER_HUB_INC
