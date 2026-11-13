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

#include <setjmp.h>
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
    ROCPROFILER_RANGE_REPLAY_STATUS_CODE_OBJECT_CHANGED_IN_RANGE,  ///< A code object was unloaded,
                                                                   ///< or a kernel from one loaded
                                                                   ///< after the range began was
                                                                   ///< dispatched, inside the range
    ROCPROFILER_RANGE_REPLAY_STATUS_HOST_JMP_NOT_SET,  ///< Tool asked for host replay, but the
                                                       ///< application did not arm a sigsetjmp in
                                                       ///< the caller's frame
    ROCPROFILER_RANGE_REPLAY_STATUS_PACKET_REPLAY_UNSUPPORTED,  ///< The tool did not ask for host
                                                                ///< replay. This build does not
                                                                ///< re-submit recorded packets, so
                                                                ///< the range closes after the
                                                                ///< application's own run
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
 * before the application observes its completion, a range covers a phase the application has
 * already started. This build replays that phase by re-running the application's own code. It does
 * not record packets and re-submit them. Pass 0 is the application's own run. Passes 1..N-1 are
 * further entries to the same function, reached by @c siglongjmp back to the @c sigsetjmp taken by
 * @ref ROCPROFILER_RANGE_REPLAY_HOST_BEGIN, after device memory and registered host memory have
 * been restored.
 *
 * A range whose tool does not set @c host_replay closes after pass 0 with
 * @ref ROCPROFILER_RANGE_REPLAY_STATUS_PACKET_REPLAY_UNSUPPORTED.
 *
 * @warning Beta. Host replay (@c host_replay set during CONFIG, range opened with
 * @ref ROCPROFILER_RANGE_REPLAY_HOST_BEGIN) re-runs the code between begin and end, so memory
 * copies, memsets, multiple queues and agents, HIP graphs, balanced allocate/free pairs, and host
 * computation all take part in every pass. Device memory on every GPU agent and the host regions
 * registered with @ref rocprofiler_range_replay_host_track_memory are restored before each extra
 * pass. Another thread dispatching or copying to a device inside the range, or a device allocation
 * left allocated or freed across the range, still declines. Module variables of a code object
 * first loaded inside the range are not rewound, so launch each kernel once before the range.
 * The application must synchronize all GPU work on every participating device before begin and
 * before end on every pass, including asynchronous copies and non-compute queues. The SDK's
 * compute-queue barriers and dispatch-handler drains do not wait for untracked copy completions.
 * Repeatability rests on the same device-memory snapshot kernel replay uses (coarse-grained
 * device allocations owned by the agent plus module-scope @c __device__ / @c __constant__
 * variables), so unified/managed memory, host memory that was not registered, and
 * virtual-memory-mapped allocations are not restored between passes.
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
    int                               host_replay;

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
    /// recorded. For host replay, the agent of the first dispatch (the range may span several).
    ///
    /// @var dispatch_count
    /// @brief [PASS, CLOSE] Number of dispatches recorded in the range. For host replay, the number
    /// of dispatches the application's own execution of the range submitted.
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
    ///
    /// @var host_replay
    /// @brief [CONFIG] Set non-zero during CONFIG @ref ROCPROFILER_CALLBACK_PHASE_ENTER to re-run
    /// the host code inside the range. Requires the application to open the range with
    /// @ref ROCPROFILER_RANGE_REPLAY_HOST_BEGIN; otherwise the range declines with
    /// @ref ROCPROFILER_RANGE_REPLAY_STATUS_HOST_JMP_NOT_SET. Zero (the default) does not replay:
    /// @ref rocprofiler_range_replay_end closes after the application's own run with
    /// @ref ROCPROFILER_RANGE_REPLAY_STATUS_PACKET_REPLAY_UNSUPPORTED, unless an earlier decline
    /// applies.
} rocprofiler_callback_tracing_range_replay_data_t;

/**
 * @brief Open a replay range on the calling thread.
 *
 * This is the one-run entry. The calling thread's dispatches until
 * @ref rocprofiler_range_replay_end are observed, and that end closes the range after the
 * application's own run. It does not raise @ref ROCPROFILER_RANGE_REPLAY_PASS. Supported replay
 * is @ref ROCPROFILER_RANGE_REPLAY_HOST_BEGIN. Ranges are thread-scoped and do not nest.
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
 * @brief Jump buffer used by @ref ROCPROFILER_RANGE_REPLAY_HOST_BEGIN.
 *
 * Returns a pointer to a thread-local @c sigjmp_buf, typed @c void* so that this header does not
 * require POSIX declarations from a strict ISO C translation unit. Calling this function arms host
 * replay for the next @ref rocprofiler_range_replay_begin on this thread, which consumes the arming
 * whether or not host replay is then used, so a buffer from an earlier range is never a jump
 * target. The buffer is only a valid target when @c sigsetjmp was called on it from a frame that
 * is still live at @ref rocprofiler_range_replay_end, which is what the macro guarantees.
 */
void*
rocprofiler_range_replay_host_jmpbuf(void) ROCPROFILER_API;

/**
 * @brief Open a replay range that re-runs the host code between it and
 * @ref rocprofiler_range_replay_end.
 *
 * Expands to a statement, not an expression: ISO C only allows @c sigsetjmp as the whole
 * controlling expression of a selection statement (or compared against a constant there), so the
 * jump target is taken in an @c if inside the caller's frame. @p status_out receives the status of
 * @ref rocprofiler_range_replay_begin on the first entry and @ref ROCPROFILER_STATUS_SUCCESS on
 * every re-entry, so CONFIG is delivered once per range. Requires @c <setjmp.h> with POSIX
 * declarations (@c sigsetjmp) in the calling translation unit.
 *
 * Contract for the code between this macro and @ref rocprofiler_range_replay_end:
 *  - Same thread, same function: the frame that expanded this macro must still be live at end.
 *  - No C++ automatic object with a non-trivial destructor may be live across the jump (declare
 *    such objects inside a nested block that closes before end, or outside the range).
 *  - Locals of the enclosing function that the range modifies must be @c volatile, or must be
 *    re-initialized inside the range; their values after the jump are otherwise indeterminate.
 *  - Host memory the range reads and writes must be registered with
 *    @ref rocprofiler_range_replay_host_track_memory to be rewound between passes.
 *  - No other thread may submit GPU work or copy into device memory while the range is open.
 *  - Synchronize all GPU work, including asynchronous copies and non-compute queues, before this
 *    macro and before end on every pass, using the runtime's supported completion mechanisms
 *    (for example, HIP device synchronization on every participating device).
 */
#define ROCPROFILER_RANGE_REPLAY_HOST_BEGIN(range_id, status_out)                                  \
    do                                                                                             \
    {                                                                                              \
        if(sigsetjmp(*(sigjmp_buf*) rocprofiler_range_replay_host_jmpbuf(), 1) == 0)               \
            (status_out) = rocprofiler_range_replay_begin(range_id);                               \
        else                                                                                       \
            (status_out) = ROCPROFILER_STATUS_SUCCESS;                                             \
    } while(0)

/**
 * @brief Register host memory that a host-replay range reads and writes.
 *
 * The contents of [@p ptr, @p ptr + @p size) are captured and copied back before each re-executed
 * pass of a host-replay range, and are hashed for the divergence check. Registration made while no
 * range is open applies to the next range opened on this thread (captured at its
 * @ref rocprofiler_range_replay_begin). Registration made during the application's own execution
 * of a host-replay range captures the region at the call. Calls made during re-executed passes, and
 * calls made in a packet-replay range, are accepted and ignored. Registrations last until the range
 * closes. The memory must stay valid until then. Where registered regions overlap, each byte is
 * restored from the earliest capture that covers it.
 *
 * @param [in] ptr  Start of the region
 * @param [in] size Size of the region in bytes
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS The region is registered (or the call was ignored)
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT @p ptr is NULL, @p size is 0, or the region
 * wraps past the end of the address space
 */
rocprofiler_status_t
rocprofiler_range_replay_host_track_memory(void* ptr, uint64_t size) ROCPROFILER_API;

/**
 * @brief Pass of the open range that the calling thread is executing.
 *
 * 0 during the application's own execution of the range and outside a range, 1..N-1 during the
 * re-executed passes of a host-replay range. Host code inside the range uses this to keep side
 * effects that must happen once (file output, logging, network I/O) out of the re-executed passes.
 *
 * @param [out] pass Pass index
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS @p pass was written
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT @p pass is NULL
 */
rocprofiler_status_t
rocprofiler_range_replay_get_host_pass(uint64_t* pass) ROCPROFILER_API;

/**
 * @brief Close the replay range open on the calling thread.
 *
 * A range opened with @ref rocprofiler_range_replay_begin runs once. This function raises
 * @ref ROCPROFILER_RANGE_REPLAY_CLOSE and returns. It does not raise
 * @ref ROCPROFILER_RANGE_REPLAY_PASS and it does not re-submit recorded packets. CLOSE status is
 * @ref ROCPROFILER_RANGE_REPLAY_STATUS_PACKET_REPLAY_UNSUPPORTED when the tool asked for more than
 * one pass, @ref ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT when it opted out, or an earlier
 * decline when one was recorded.
 *
 * For a host-replay range this function does not return after passes 0..N-2: it waits for every
 * GPU queue to go idle, restores device memory and the registered host regions, raises
 * @ref ROCPROFILER_RANGE_REPLAY_PASS, and calls @c siglongjmp back to
 * @ref ROCPROFILER_RANGE_REPLAY_HOST_BEGIN. It returns once, after the last pass, having raised
 * CLOSE. The final pass's device and host state is what the application continues with.
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
