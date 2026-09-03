// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup CALLBACK_TRACING_SERVICE Synchronous Tracing Services
 * @brief Experimental APIs
 *
 * @{
 */

/**
 * @brief Outcome of a range replay attempt.
 *
 * Delivered on @ref ROCPROFILER_RANGE_REPLAY_CLOSE so a tool can tell a replayed range from a
 * declined one and report why. Range replay declines rather than aborts: the application's own
 * (live) execution of the range is never altered by a decline, only the extra passes are skipped.
 */
typedef enum rocprofiler_range_replay_status_t  // NOLINT(performance-enum-size)
{
    ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED = 0,   ///< The range was replayed
    ROCPROFILER_RANGE_REPLAY_STATUS_NO_DISPATCH,    ///< No dispatch was recorded in the range
    ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT,  ///< Tool asked for <= 1 pass (opt-out)
    ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE,    ///< Range spanned more than one HSA queue
    ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_AGENT,    ///< Range spanned more than one agent
    ROCPROFILER_RANGE_REPLAY_STATUS_GRAPH_LAUNCH,   ///< Range contained a HIP graph launch
    ROCPROFILER_RANGE_REPLAY_STATUS_UNKNOWN_KERNARG_SIZE,  ///< A kernel's kernarg size was not
                                                           ///< resolvable at record time
    ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE,  ///< An async copy touched device memory
                                                           ///< inside the range
    ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH,  ///< Another thread dispatched to the same
                                                          ///< agent inside the range
    ROCPROFILER_RANGE_REPLAY_STATUS_PROGRAM_TOO_LARGE,    ///< Range exceeded the record budget
    ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED,      ///< Device-memory snapshot was incomplete
    ROCPROFILER_RANGE_REPLAY_STATUS_STAGING_FAILED,  ///< Kernarg staging or signal setup failed
    ROCPROFILER_RANGE_REPLAY_STATUS_ALLOCATION_CHANGED_IN_RANGE,  ///< A device allocation was made
                                                                  ///< or freed inside the range
    ROCPROFILER_RANGE_REPLAY_STATUS_UNSUPPORTED_QUEUE_PATH,  ///< The queue submission path in use
                                                             ///< cannot re-submit recorded packets
    ROCPROFILER_RANGE_REPLAY_STATUS_LAST,
} rocprofiler_range_replay_status_t;

/**
 * @brief ROCProfiler Range Replay Callback Tracer Record.
 *
 * Payload for @ref ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY callbacks. All members are present in
 * the struct (no unions); which members are meaningful depends on the operation:
 *
 * - @ref ROCPROFILER_RANGE_REPLAY_CONFIG: delivered from @ref rocprofiler_range_replay_begin.
 *   @c range_id is populated by the SDK. The tool sets @c pass_count_cb and optionally
 *   @c replay_continue_cb during @ref ROCPROFILER_CALLBACK_PHASE_ENTER.
 * - @ref ROCPROFILER_RANGE_REPLAY_PASS: @c range_id, @c current_pass, @c total_passes,
 *   @c agent_id and @c dispatch_count are populated. @c current_pass is 1-based here: pass 0 is
 *   the application's own execution of the range, which the SDK observes rather than drives, so
 *   only re-executed passes raise a PASS callback.
 * - @ref ROCPROFILER_RANGE_REPLAY_CLOSE: delivered from @ref rocprofiler_range_replay_end after
 *   the last pass. @c status says whether the range was replayed or why it was declined, and
 *   @c divergence_count reports how many snapshot regions differed between the application's
 *   execution and the final replayed pass (0 when verification is off or nothing diverged).
 *
 * Unlike @ref ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY, which re-executes one dispatch in place
 * before the application observes its completion, a range covers a *sequence* of dispatches that
 * the application has already observed. The SDK therefore records the range while it runs, and
 * re-executes the recording afterwards: pass 0 is the application's own run, passes 1..N-1 are
 * SDK-driven re-executions of the recorded dispatches from the range-entry memory state.
 *
 * @warning Beta. A range is replayed only when every recorded dispatch targets one queue on one
 * agent, no HIP graph launch occurs inside it, no async copy writes device memory inside it, no
 * device allocation is made or freed inside it, and no other thread dispatches to the same agent
 * while it is open. Repeatability rests on the same
 * device-memory snapshot kernel replay uses (coarse-grained device allocations owned by the agent
 * plus module-scope @c __device__ / @c __constant__ variables), so unified/managed memory, host and
 * fine-grained memory, and virtual-memory-mapped allocations are not restored between passes.
 * Replayed passes are serialized dispatch-by-dispatch, so concurrency-sensitive measurements differ
 * from pass 0. Host state is not rewound at all: a range whose kernels consume values the host
 * recomputed inside the range is not a candidate for replay.
 *
 * @see `source/docs/conceptual/range_replay/index.md`
 */
typedef struct rocprofiler_callback_tracing_range_replay_data_t
{
    uint64_t size;      ///< sizeof this struct (versioning)
    uint64_t range_id;  ///< tool-supplied id passed to @ref rocprofiler_range_replay_begin

    uint64_t (*pass_count_cb)(uint64_t range_id, rocprofiler_user_data_t user_data);

    int (*replay_continue_cb)(uint64_t                range_id,
                              uint64_t                current_pass,
                              uint64_t                total_passes,
                              rocprofiler_user_data_t user_data);

    uint64_t current_pass;
    uint64_t total_passes;

    rocprofiler_status_t (*replay_local_start_context_cb)(rocprofiler_context_id_t context_id);
    rocprofiler_status_t (*replay_local_stop_context_cb)(rocprofiler_context_id_t context_id);

    rocprofiler_agent_id_t            agent_id;
    uint64_t                          dispatch_count;
    rocprofiler_range_replay_status_t status;
    uint64_t                          divergence_count;

    /// @var pass_count_cb
    /// @brief [CONFIG] Tool-provided callback returning the total number of passes for this range,
    /// counting the application's own execution as pass 0. The tool sets this during CONFIG
    /// @ref ROCPROFILER_CALLBACK_PHASE_ENTER; the SDK then calls it (if non-null) to size the loop:
    ///  - left NULL     => the range is observed but never re-executed (per-range opt-out)
    ///  - returns N > 1 => the recorded range is re-executed N-1 times after the live run
    ///  - returns 0     => open-ended re-execution (requires @c replay_continue_cb)
    ///  - returns 1     => opt-out, same as NULL
    ///
    /// @var replay_continue_cb
    /// @brief [CONFIG] Optional callback invoked after each re-executed pass. Return non-zero to
    /// keep going, zero to stop. Required when @c pass_count_cb returns 0.
    ///
    /// @var current_pass
    /// @brief [PASS] 1-based index of the re-executed pass (pass 0 is the application's run).
    ///
    /// @var total_passes
    /// @brief [PASS] Total passes if known (the value @c pass_count_cb returned), else 0.
    ///
    /// @var replay_local_start_context_cb
    /// @var replay_local_stop_context_cb
    /// @brief [PASS] Localized context control, with the same semantics as the kernel replay
    /// toggles: valid only during PASS @ref ROCPROFILER_CALLBACK_PHASE_ENTER, sticky across passes
    /// of this range, scoped to the range's replay loop, and unable to promote a context that is
    /// globally inactive. @see rocprofiler-sdk/experimental/kernel_replay.h
    ///
    /// @var agent_id
    /// @brief [PASS, CLOSE] Agent the range was bound to, or a zero handle when no dispatch was
    /// recorded.
    ///
    /// @var dispatch_count
    /// @brief [PASS, CLOSE] Number of dispatches recorded in the range.
    ///
    /// @var status
    /// @brief [CLOSE] Replay outcome. @ref ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED when at least
    /// one extra pass ran, otherwise the reason the range was declined.
    ///
    /// @var divergence_count
    /// @brief [CLOSE] Number of snapshot regions whose contents differed between the end of the
    /// application's execution of the range and the end of the final replayed pass. A non-zero
    /// value means the range is not self-contained under the snapshot's coverage, so per-pass
    /// measurements describe different inputs.
} rocprofiler_callback_tracing_range_replay_data_t;

/**
 * @brief Open a replay range on the calling thread.
 *
 * The dispatches the calling thread submits until the matching @ref rocprofiler_range_replay_end
 * are recorded, and a device-memory snapshot of the range's agent is taken before the first of them
 * executes. Ranges are thread-scoped and do not nest.
 *
 * Emits @ref ROCPROFILER_RANGE_REPLAY_CONFIG so the tool can supply the pass count.
 *
 * @param [in] range_id Tool-defined identifier echoed back in every callback for this range.
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS A range was opened
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND No context configured a range replay service
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED No such context is active
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT A range is already open on this thread
 */
rocprofiler_status_t
rocprofiler_range_replay_begin(uint64_t range_id) ROCPROFILER_API;

/**
 * @brief Close the replay range open on the calling thread and run its extra passes.
 *
 * Drains the agent, then re-executes the recorded dispatches from the snapshot taken at
 * @ref rocprofiler_range_replay_begin, raising @ref ROCPROFILER_RANGE_REPLAY_PASS around each pass
 * and @ref ROCPROFILER_RANGE_REPLAY_CLOSE once at the end. Blocks the calling thread for the
 * duration of the replay.
 *
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS The range was closed (replayed or declined; see the CLOSE
 * callback's @c status)
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT No range is open on this thread
 */
rocprofiler_status_t
rocprofiler_range_replay_end(void) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
