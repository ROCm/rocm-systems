/**
 * @file profiler_hub_trim.h
 * @brief Trimmer API for libprofiler-hub.
 *
 * Separate from the reader API (profiler_hub.h) and the writer API
 * (profiler-hub/cpp/writer.hpp). A trace is opened with the reader, then
 * trimmed through ph_trim_trace().
 */

#ifndef PROFILER_HUB_TRIM_INC
#define PROFILER_HUB_TRIM_INC

#include "profiler_hub_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Writes a trimmed copy of the trace opened in @p ctx.
     * @param start_ts Inclusive overlap start, in absolute trace nanoseconds.
     *        The caller adds the trace origin before calling.
     * @param end_ts Exclusive overlap end, in absolute trace nanoseconds.
     * @param output_path File to create. An existing file at this path is
     *        replaced. It must not be the source trace.
     * @param future Null to trim before returning. Otherwise a future from
     *        ph_future_create() that was not passed to an operation yet.
     * @return PH_RESULT_SUCCESS on success, PH_RESULT_INVALID_CONTEXT if
     *         @p ctx is null or the trace failed to load,
     *         PH_RESULT_INVALID_ARGUMENT if @p output_path is null or names
     *         the source file, PH_RESULT_INVALID_FILE if the output cannot be
     *         replaced, PH_RESULT_UNSUPPORTED_FILE if this schema has no trim
     *         rules or the files in @p ctx differ in schema,
     *         PH_RESULT_CANCELLED if @p future is cancelled.
     * @note The copy keeps the source schema. Duration rows that overlap the
     *       window are kept, and rows that belong to those events are kept
     *       with them. Tables the schema does not describe are copied whole.
     */
    ph_result_t ph_trim_trace(ph_ctx_t    ctx,
                              uint64_t    start_ts,
                              uint64_t    end_ts,
                              const char* output_path,
                              ph_future_t future);

#ifdef __cplusplus
}
#endif

#endif  // #ifndef PROFILER_HUB_TRIM_INC
