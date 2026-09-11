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
     * @brief Not yet implemented.
     * @warning Declared for the planned async task API but has no
     *          definition in the current version of the library; calling it
     *          will fail to link.
     */
    ph_result_t ph_future_get(ph_ctx_t     ctx,
                              ph_future_t* future,
                              ph_task_fn   task_fn,
                              void*        user_data);

    /**
     * @brief Not yet implemented.
     * @warning See ph_future_get().
     */
    ph_result_t ph_future_wait(ph_ctx_t ctx, ph_future_t future);

    /**
     * @brief Not yet implemented.
     * @warning See ph_future_get().
     */
    ph_result_t ph_future_cancel(ph_ctx_t ctx, ph_future_t future);

    /**
     * @brief Not yet implemented.
     * @warning See ph_future_get().
     */
    ph_result_t ph_future_free(ph_ctx_t ctx, ph_future_t future);

#ifdef __cplusplus
}
#endif

#endif  // #ifndef PROFILER_HUB_INC
