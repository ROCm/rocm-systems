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
    ROCPROFILER_RANGE_REPLAY_STATUS_BODY_EXCEPTION,    ///< The body passed to
                                                       ///< rocprofiler_range_replay_host_run exited
                                                       ///< by an exception before the last pass
    ROCPROFILER_RANGE_REPLAY_STATUS_LAST,
} rocprofiler_range_replay_status_t;

/**
 * @brief ROCProfiler Range Replay Callback Tracer Record.
 *
 * Payload for @ref ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY callbacks. All members are present in
 * the struct (no unions); which members are meaningful depends on the operation:
 *
 * - @ref ROCPROFILER_RANGE_REPLAY_CONFIG: delivered from @ref rocprofiler_range_replay_begin (or
 *   once from @ref rocprofiler_range_replay_host_run). @c range_id is populated by the SDK. The
 * tool sets @c pass_count_cb and optionally
 *   @c replay_continue_cb during @ref ROCPROFILER_CALLBACK_PHASE_ENTER.
 * - @ref ROCPROFILER_RANGE_REPLAY_PASS: @c range_id, @c current_pass, @c total_passes,
 *   @c agent_id and @c dispatch_count are populated. @c current_pass is 1-based here: pass 0 is
 *   the application's own execution of the range, which the SDK observes rather than drives, so
 *   only re-executed passes raise a PASS callback.
 * - @ref ROCPROFILER_RANGE_REPLAY_CLOSE: delivered from @ref rocprofiler_range_replay_end (or from
 *   @ref rocprofiler_range_replay_host_run) after the last pass. @c status says whether the range
 * was replayed or why it was declined, and
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
 * from pass 0. A range whose kernels consume values the host recomputed inside the range is not a
 * candidate for packet replay.
 *
 * Host replay (@c host_replay set during CONFIG, range opened with
 * @ref ROCPROFILER_RANGE_REPLAY_HOST_BEGIN or run with @ref rocprofiler_range_replay_host_run)
 * lifts the packet restrictions: nothing is recorded, and
 * each extra pass re-runs the application's own code between begin and end, so memory copies,
 * memsets, multiple queues and agents, HIP graphs, balanced allocate/free pairs, and host
 * computation all take part in every pass. Device memory on every GPU agent and the host regions
 * registered with @ref rocprofiler_range_replay_host_track_memory are restored before each extra
 * pass. Another thread dispatching or copying to a device inside the range, or a device allocation
 * left allocated or freed across the range, still declines. Module variables of a code object
 * first loaded inside the range are not rewound, so launch each kernel once before the range.
 * The application must synchronize all GPU work on every participating device before begin and
 * before end on every pass, including asynchronous copies and non-compute queues. The SDK's
 * compute-queue barriers and dispatch-handler drains do not wait for untracked copy completions.
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
    /// the host code inside the range instead of re-submitting recorded packets. Requires the
    /// application to open the range with @ref ROCPROFILER_RANGE_REPLAY_HOST_BEGIN or run it with
    /// @ref rocprofiler_range_replay_host_run; otherwise the range declines with
    /// @ref ROCPROFILER_RANGE_REPLAY_STATUS_HOST_JMP_NOT_SET. Zero (the default) keeps packet
    /// replay.
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
 * @brief Close the replay range open on the calling thread and run its extra passes.
 *
 * Drains the agent, then re-executes the recorded dispatches from the snapshot taken at
 * @ref rocprofiler_range_replay_begin, raising @ref ROCPROFILER_RANGE_REPLAY_PASS around each pass
 * and @ref ROCPROFILER_RANGE_REPLAY_CLOSE once at the end. Blocks the calling thread for the
 * duration of the replay.
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
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT No range is open on this thread, or the open
 * range belongs to @ref rocprofiler_range_replay_host_run (the call came from its body); that range
 * is left open and unchanged
 */
rocprofiler_status_t
rocprofiler_range_replay_end(void) ROCPROFILER_API;

/**
 * @brief Body of a range run by @ref rocprofiler_range_replay_host_run.
 *
 * @param [in] pass      Pass being executed: 0 for the application's own execution of the range,
 *                       1..N-1 for re-executed passes. The value
 *                       @ref rocprofiler_range_replay_get_host_pass reports inside the body.
 * @param [in] user_data The @p user_data given to @ref rocprofiler_range_replay_host_run
 */
typedef void (*rocprofiler_range_replay_body_t)(uint64_t pass, void* user_data);

/**
 * @brief Run a replay range whose body is a function the SDK calls once per pass.
 *
 * The function-call form of @ref ROCPROFILER_RANGE_REPLAY_HOST_BEGIN and
 * @ref rocprofiler_range_replay_end. The range is opened as by @ref rocprofiler_range_replay_begin,
 * so CONFIG is delivered once; @p body is called; the range is closed as by
 * @ref rocprofiler_range_replay_end; and the function returns after CLOSE. There is no @c sigsetjmp
 * or @c siglongjmp on this path: every pass is an ordinary call that returns, so C++ objects in the
 * body are destroyed at the end of each pass, locals need not be @c volatile, and any language that
 * can pass a C function pointer can use it. The cost over the macro is one indirect call per pass.
 *
 * The tool chooses the replay at CONFIG:
 *  - @c host_replay set: host replay. After each return of @p body the SDK fences the intercepted
 *    compute queues on every GPU agent; while passes remain it restores device memory and the
 *    registered host regions, raises @ref ROCPROFILER_RANGE_REPLAY_PASS, and calls @p body again
 *    with the next pass. @ref ROCPROFILER_RANGE_REPLAY_STATUS_HOST_JMP_NOT_SET is never reported.
 *  - @c host_replay unset: packet replay. @p body is called once, with pass 0, and the dispatches
 * it submits are recorded and re-executed exactly as between begin and end.
 *
 * When the range cannot be opened, @p body still runs exactly once, so the application's work
 * happens whether or not a tool replays it. It is passed the pass the calling thread is executing:
 * 0, or the current pass of an enclosing host-replay range. Host regions it registers while no
 * range is open are dropped when this function returns rather than carried to the next range.
 *
 * Contract for @p body (the contract between begin and end, without the jump rules):
 *  - All GPU work it submits, including asynchronous copies and non-compute queues, must be
 * complete when it returns, and GPU work submitted before this call must be complete before it (for
 *    example, HIP device synchronization on every participating device). The SDK's compute-queue
 *    fences do not wait for untracked copy completions.
 *  - Host memory it reads and writes across passes must be registered with
 *    @ref rocprofiler_range_replay_host_track_memory, before this call or from @p body on pass 0,
 * to be rewound between passes. @ref rocprofiler_range_replay_get_host_pass and
 *    @ref rocprofiler_range_replay_host_track_memory behave inside @p body as between begin and
 * end.
 *  - No other thread may submit GPU work or copy into device memory while it runs.
 *  - Ranges do not nest. From @p body, @ref rocprofiler_range_replay_begin,
 *    @ref rocprofiler_range_replay_end and this function return
 *    @ref ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT and leave the running range unchanged; a nested
 *    call of this function still runs its own body once.
 *  - It must return or throw a C++ exception. Leaving it with @c longjmp leaves the range open on
 * the thread.
 *
 * If @p body throws, the range closes without the passes it has left: the pass's GPU work is
 * fenced, PASS exit is raised for a re-executed pass, and CLOSE reports
 * @ref ROCPROFILER_RANGE_REPLAY_STATUS_BODY_EXCEPTION (or the reason the range had already been
 * declined for). The exception then propagates to the caller. Nothing is restored: the application
 * continues from the state the throwing pass left. This requires the caller and @p body to be
 * compiled with C++ exceptions; see @c rocprofiler::range_replay::host_run.
 *
 * @param [in] range_id  Tool-defined identifier echoed back in every callback for this range.
 * @param [in] body      Called once per pass
 * @param [in] user_data Passed to every call of @p body
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS The range ran and was closed (replayed or declined; see the
 * CLOSE callback's @c status)
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND No context configured a range replay
 * service; @p body ran once
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED No such context is active; @p body ran
 * once
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT @p body is NULL (nothing ran and no state
 * changed), or a range is already open on this thread (@p body ran once)
 */
rocprofiler_status_t
rocprofiler_range_replay_host_run(uint64_t                        range_id,
                                  rocprofiler_range_replay_body_t body,
                                  void*                           user_data) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI

#if defined(__cplusplus) && __cplusplus >= 201703L
#    include <memory>
#    include <type_traits>

namespace rocprofiler
{
namespace range_replay
{
/**
 * @brief C++ form of @ref rocprofiler_range_replay_host_run.
 *
 * @p body is any callable, typically a lambda with captures. It is invoked as @c body(pass) when
 * it accepts a @c uint64_t, otherwise as @c body(), and its result is discarded. It is called by
 * reference through @c user_data, so it is neither copied nor moved: state it keeps (a mutable
 * lambda's captures) is shared by every pass and, like any host memory, is rewound only if
 * registered with @ref rocprofiler_range_replay_host_track_memory. An exception it throws closes
 * the range and propagates out of this function. Requires C++17.
 */
template <typename FuncT>
inline rocprofiler_status_t
host_run(uint64_t range_id, FuncT&& body)
{
    using body_ptr_t = std::add_pointer_t<std::remove_reference_t<FuncT>>;

    body_ptr_t callable = std::addressof(body);
    return rocprofiler_range_replay_host_run(
        range_id,
        [](uint64_t pass, void* user_data) {
            auto* func = *static_cast<body_ptr_t*>(user_data);
            if constexpr(std::is_invocable_v<decltype(*func), uint64_t>)
                (*func)(pass);
            else
                (*func)();
        },
        &callable);
}
}  // namespace range_replay
}  // namespace rocprofiler
#endif
