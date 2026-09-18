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
     * @param file_path Path to the trace database to open.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_CONTEXT_ALLOCATION_FAILED if the
     *         trace could not be opened/parsed.
     * @note On success, the caller owns @p *ctx and must release it with
     *       ph_ctx_free().
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
     *       point into memory owned by @p ctx. They remain valid only until
     *       @p ctx is freed or another call to this function rebuilds the
     *       track list.
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
     * @param track_id Id of a track with ph_track_t::agent_id == 0, as
     *        returned by ph_get_track_list()/ph_get_node().
     * @param start_ts Start of the time window (ns), or 0 for no filter.
     * @param end_ts End of the time window (ns), or 0 for no filter.
     * @param events Out parameter receiving the event list.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p events is
     *         null or @p track_id does not identify a known track.
     * @note @p events->events and every ph_event_t::name in it point into
     *       memory owned by @p ctx and remain valid until @p ctx is freed.
     *       Unlike ph_get_track_list()/ph_get_node(), a later call to this
     *       function does NOT invalidate an earlier one's result -- each
     *       call gets its own private storage, safe to read concurrently
     *       from multiple calls (including calls made via ph_future_get()).
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
     * @param track_id Id of a track with ph_track_t::agent_id != 0, as
     *        returned by ph_get_track_list()/ph_get_node().
     * @param start_ts Start of the time window (ns), or 0 for no filter.
     * @param end_ts End of the time window (ns), or 0 for no filter.
     * @param samples Out parameter receiving the sample list.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p samples is
     *         null or @p track_id does not identify a known track.
     * @note @p samples->samples points into memory owned by @p ctx and
     *       remains valid until @p ctx is freed. Unlike
     *       ph_get_track_list()/ph_get_node(), a later call to this
     *       function does NOT invalidate an earlier one's result -- each
     *       call gets its own private storage, safe to read concurrently
     *       from multiple calls (including calls made via ph_future_get()).
     */
    ph_result_t ph_get_track_samples(ph_ctx_t          ctx,
                                     uint32_t          track_id,
                                     uint64_t          start_ts,
                                     uint64_t          end_ts,
                                     ph_sample_list_t* samples);

    /**
     * @brief Submits @p task_fn for asynchronous execution on @p ctx's
     *        internal thread pool.
     * @param ctx Context to submit the task to. Owns the returned future;
     *        it is cancelled and waited on if still live when @p ctx is
     *        freed.
     * @param future Out parameter receiving the new future handle. Must
     *        not be null.
     * @param task_fn Callback invoked with @p user_data on a worker
     *        thread. Must write any result/error into memory owned by
     *        @p user_data; this API has no return-value/error channel.
     * @param user_data Passed through to @p task_fn unchanged.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p future or
     *         @p task_fn is null, PH_RESULT_FUTURE_ALLOCATION_FAILED if
     *         the future could not be allocated.
     * @note Caller owns @p *future and must release it with
     *       ph_future_free().
     */
    ph_result_t ph_future_get(ph_ctx_t     ctx,
                              ph_future_t* future,
                              ph_task_fn   task_fn,
                              void*        user_data);

    /**
     * @brief Blocks until @p future's task finishes running or is
     *        cancelled.
     * @param ctx Context @p future was created through.
     * @param future Future to wait on.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p future is
     *         null or was not issued by @p ctx.
     */
    ph_result_t ph_future_wait(ph_ctx_t ctx, ph_future_t future);

    /**
     * @brief Cancels @p future's task.
     * @param ctx Context @p future was created through.
     * @param future Future to cancel.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p future is
     *         null or was not issued by @p ctx.
     * @note If the task has not started running yet, it is skipped
     *       entirely. If it is already running, cancellation is a no-op:
     *       @p task_fn's void(*)(void*) signature gives it no way to
     *       observe a stop request, so it always runs to completion once
     *       started.
     */
    ph_result_t ph_future_cancel(ph_ctx_t ctx, ph_future_t future);

    /**
     * @brief Releases @p future.
     * @param ctx Context @p future was created through.
     * @param future Future to release.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null, PH_RESULT_INVALID_ARGUMENT if @p future is
     *         null or was not issued by @p ctx.
     * @warning Does not wait for the task to finish; call
     *          ph_future_wait() first if that is required.
     */
    ph_result_t ph_future_free(ph_ctx_t ctx, ph_future_t future);

#ifdef __cplusplus
}
#endif

#endif  // #ifndef PROFILER_HUB_INC
