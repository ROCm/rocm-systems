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
 * @brief Tool-provided callback returning the number of replay passes for a dispatch.
 */
typedef uint64_t (*rocprofiler_kernel_replay_pass_count_cb_t)(
    rocprofiler_kernel_dispatch_info_t dispatch_info,
    rocprofiler_user_data_t            user_data);

/**
 * @brief Tool-provided callback invoked after each pass to decide whether the loop continues.
 */
typedef int (*rocprofiler_kernel_replay_continue_cb_t)(
    rocprofiler_kernel_dispatch_info_t dispatch_info,
    uint64_t                           current_pass,
    uint64_t                           total_passes,
    rocprofiler_user_data_t            user_data);

/**
 * @brief SDK-provided callback marking a context enabled or disabled for the current replay loop.
 */
typedef rocprofiler_status_t (*rocprofiler_kernel_replay_context_cb_t)(
    rocprofiler_context_id_t context_id);

/**
 * @brief Bits of @ref rocprofiler_kernel_replay_options_t::flags.
 */
typedef enum rocprofiler_kernel_replay_option_flag_t
{
    ROCPROFILER_KERNEL_REPLAY_OPTION_NONE                   = 0,         ///< The SDK's defaults
    ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_DEVICE        = (1 << 0),  ///< Device memory
    ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_HOST_PINNED   = (1 << 1),  ///< Pinned host memory
    ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_HOST_PAGEABLE = (1 << 2),  ///< Not supported yet
    ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_DISK          = (1 << 3),  ///< Not supported yet
} rocprofiler_kernel_replay_option_flag_t;

/**
 * @brief Options a tool sets for one dispatch during CONFIG @ref ROCPROFILER_CALLBACK_PHASE_ENTER.
 *
 * The snapshot bits of @c flags name the memory the SDK may keep the dispatch's snapshot in:
 *  - only @ref ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_HOST_PINNED: the snapshot is kept in
 *    pinned host memory, so it does not compete with an application that allocates device memory
 *    as it runs, apart from a work list the restore kernels keep in device memory between replays:
 *    about 0.2% of the snapshot's size, and at least 2 MiB. Pinned memory cannot be swapped out,
 *    so a large snapshot leaves the rest of the system that much less memory.
 *  - only @ref ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_DEVICE: the snapshot is kept in device
 *    memory. The SDK leaves the application 5% of the free device memory, counting memory it kept
 *    from earlier replays as free, because the replayed kernel may need scratch. This needs the
 *    SDK's copy kernels for the GPU.
 *  - neither, or both of those: the SDK chooses, which today means device memory first and pinned
 *    host memory for what does not fit, within limits it sets for each, and keeps that memory for
 *    later replays.
 *
 * @ref ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_HOST_PAGEABLE and
 * @ref ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_DISK name pageable host memory and a file on disk.
 * This SDK cannot keep a snapshot in either yet, so options that set one of them are not honored.
 *
 * With one flag, the SDK sets no limit of its own on how much of that memory the snapshot uses,
 * beyond the device reserve above. It frees that memory when the replay ends, including memory it
 * kept there from earlier replays, so a tool that switches flags between dispatches makes it
 * allocate again. A replay limited to pinned host memory also frees device memory kept by earlier
 * replays. If the snapshot does not fit in the memory allowed, the dispatch is not replayed: it
 * runs once, as when @c replay_pass_count returns 1.
 *
 * Members added later come with a flag bit that enables them, so an SDK that predates a member
 * rejects a tool that uses it instead of ignoring it.
 */
typedef struct rocprofiler_kernel_replay_options_t
{
    uint64_t size;   ///< sizeof(rocprofiler_kernel_replay_options_t)
    uint64_t flags;  ///< Bitwise OR of @ref rocprofiler_kernel_replay_option_flag_t values
} rocprofiler_kernel_replay_options_t;

/**
 * @brief ROCProfiler Kernel Replay Callback Tracer Record.
 *
 * Payload for @ref ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY callbacks.
 * All members are present in the struct (no unions). Which members are meaningful
 * depends on the current operation:
 *
 * - @ref ROCPROFILER_KERNEL_REPLAY_CONFIG: @c dispatch_info is populated by the SDK.
 *   The tool sets @c replay_pass_count and optionally @c replay_continue and @c replay_options
 *   during @ref ROCPROFILER_CALLBACK_PHASE_ENTER. Pass-info fields are zero.
 * - @ref ROCPROFILER_KERNEL_REPLAY_PASS: @c dispatch_info, @c current_pass, and
 *   @c total_passes are populated by the SDK. @c replay_pass_count and @c replay_options read as
 *   NULL and must not be modified (both are fixed at CONFIG PHASE_ENTER). @c replay_continue is
 *   NULL during PASS PHASE_ENTER; during PASS PHASE_EXIT the SDK populates it with the callback
 *   in effect for this pass, and the tool may overwrite it there to change the continue decision
 *   for this pass only (the next pass reverts to the CONFIG default).
 *
 * Tool callbacks that run while a dispatch is replayed (the PASS callbacks, and the callbacks of
 * services such as dispatch counting for the replayed passes) must not allocate or free device
 * memory. The SDK holds the allocation inventory's read lock from each restore until its pass
 * completes, so such a call waits on that lock, and the replay hangs or aborts when its drain
 * times out.
 *
 * The SDK maintains a single @c rocprofiler_user_data_t for the entire replay sequence
 * (CONFIG + all PASS operations). A tool can write per-dispatch state into
 * @c user_data during CONFIG PHASE_ENTER; the same value is delivered to every
 * subsequent PASS callback and to @c replay_pass_count and @c replay_continue for the
 * same dispatch.
 *
 * Each PASS receives its own copy of that value, so a write during a PASS callback is scoped to
 * the pass that performed it: that pass's PASS PHASE_EXIT and its @c replay_continue decision
 * observe the write, while the next PASS and CONFIG PHASE_EXIT still observe the value set during
 * CONFIG PHASE_ENTER. @c replay_continue runs after PASS PHASE_EXIT and receives that pass's copy
 * (the value its PHASE_EXIT left), not the CONFIG value. For state that must persist across the
 * whole sequence regardless of per-pass writes, place it behind @c user_data.ptr: mutating the
 * pointed-to object is visible from every callback, whereas overwriting the union value is
 * per-pass. See `samples/kernel_replay/basic_client_with_user_data.cpp` for a worked example.
 *
 * Exactly one context may configure a KERNEL_REPLAY service; a second
 * @ref rocprofiler_configure_callback_tracing_service for this domain returns
 * @ref ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED.
 *
 * @warning Beta. A dispatch is replayed only when its submission is a single packet containing a
 * single dispatch. HIP graph launches are not replayed, and a multi-packet submission runs once
 * without replay; each case warns once. Repeatability rests on a snapshot covering coarse-grained
 * device allocations owned by the agent plus module-scope @c __device__ / @c __constant__
 * variables exposed as HSA variable symbols. Internal or static device state not exposed as
 * symbols is not restored.
 *
 * Argument-memory checks are best effort: indirect pointers and some aggregate fields can be
 * missed, while scalar values can cause false declines. Detected memory outside the snapshot
 * (fine-grained device, unified or managed memory, host memory, another agent's memory,
 * @c hipMallocAsync and other virtual-memory-mapped allocations), failed snapshot capture, or
 * failed required memory checks cause the dispatch to run once without replay. This includes a
 * kernarg buffer that cannot be verified and safely inspected.
 *
 * Selected runtime copy/fill kernels bypass the guard; overlapping copies are not detected.
 * Allocations carrying @c HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG are excluded from the snapshot.
 * A failed restore aborts the process rather than continuing with partially restored memory.
 *
 * @see `docs/how-to/using-kernel-replay.rst` for the full limitation list and
 * `docs/conceptual/kernel_replay/kernel_replay_memory_snapshot.md` for what the snapshot covers
 * and why.
 */
typedef struct rocprofiler_callback_tracing_kernel_replay_data_t
{
    uint64_t                                  size;           ///< Size of this struct
    rocprofiler_kernel_dispatch_info_t        dispatch_info;  ///< Kernel dispatch info (always set)
    uint64_t                                  current_pass;
    uint64_t                                  total_passes;
    rocprofiler_kernel_replay_pass_count_cb_t replay_pass_count;
    rocprofiler_kernel_replay_continue_cb_t   replay_continue;
    rocprofiler_kernel_replay_context_cb_t    replay_start_context;
    rocprofiler_kernel_replay_context_cb_t    replay_stop_context;
    const rocprofiler_kernel_replay_options_t* replay_options;

    /// @var replay_pass_count
    /// @brief [CONFIG] Tool-provided callback returning the number of replay passes.
    /// The tool sets this during CONFIG @ref ROCPROFILER_CALLBACK_PHASE_ENTER; the SDK
    /// then calls it (if non-null) to obtain the pass count for this dispatch:
    ///  - left NULL     => dispatch is NOT replayed; it runs once and execution
    ///                     continues as usual (no snapshot, per-dispatch opt-out)
    ///  - returns 1     => dispatch is NOT replayed either: one pass needs no snapshot, so
    ///                     it takes the ordinary single-dispatch path and no PASS callbacks
    ///                     are issued (a second per-dispatch opt-out)
    ///  - returns N > 1 => fixed loop of N passes (optional @c replay_continue honored)
    ///  - returns 0     => indefinite loop (requires @c replay_continue)
    /// @c dispatch_info and @c user_data are provided so the tool can pick the count
    /// per dispatch and thread per-dispatch state through the callbacks.
    ///
    /// @var replay_continue
    /// @brief [CONFIG/PASS] Optional tool-provided callback invoked after each pass completes.
    /// Return non-zero to continue the replay loop, zero to break out.
    /// Required when @c replay_pass_count returns 0; with a fixed count N > 1 it allows early exit
    /// only — it cannot extend the loop past N. It is not consulted after the final pass of a
    /// fixed loop (for N passes it sees @c current_pass 0 through N-2); an indefinite loop consults
    /// it after every pass. @c rocprofv3 does not set this callback; adaptive or early-exit control
    /// is a custom-tool feature.
    /// Set the default during CONFIG @ref ROCPROFILER_CALLBACK_PHASE_ENTER; the PASS @ref
    /// ROCPROFILER_CALLBACK_PHASE_EXIT callback may override it for that pass only, after which the
    /// next pass reverts to the CONFIG default.
    /// @c dispatch_info and @c user_data are provided, where @c user_data is the pass-scoped copy
    /// (the value that pass's PASS @ref ROCPROFILER_CALLBACK_PHASE_EXIT left), not the
    /// sequence-wide CONFIG value.
    ///
    /// @var current_pass
    /// @brief [PASS] 0-indexed current pass number. Read-only, populated by SDK.
    ///
    /// @var total_passes
    /// @brief [PASS] Total passes if known (the value passed to @c replay_pass_count),
    /// else 0. Read-only.
    ///
    /// @var replay_start_context
    /// @var replay_stop_context
    /// @brief [PASS] Per-pass context enable/disable mask. The SDK populates these function
    /// pointers before each PASS @ref ROCPROFILER_CALLBACK_PHASE_ENTER; the tool calls them
    /// to mark an already-active context enabled or disabled for the current replay loop
    /// only. They record overrides in a thread-scoped map and do **not** invoke global
    /// @ref rocprofiler_start_context / @ref rocprofiler_stop_context. Semantics:
    ///  - Only valid to call during PASS @ref ROCPROFILER_CALLBACK_PHASE_ENTER.
    ///  - Sticky across passes: a context disabled in one pass stays disabled until it
    ///    is enabled again within the same replay loop (and vice versa). A tool therefore
    ///    positions a context once rather than re-issuing the same mask every pass.
    ///  - Scoped to the replay loop: each context's pre-replay active/inactive state
    ///    is restored once the loop completes. Global context state is never modified.
    ///  - A local enable only undoes a prior local disable; it cannot promote a context that is
    ///    globally inactive (its service/callback thread may already be stopped).
    ///  - Coverage varies by service. Kernel dispatch tracing and dispatch thread trace
    ///    observe a local stop only: they skip a dispatch whose context is forced off, and
    ///    have no means to add a context that is not already collecting. Dispatch counter
    ///    collection and SPM consult the override on every dispatch (and likewise refuse to
    ///    promote a globally stopped context). PC sampling is agent-wide and device counting
    ///    is not dispatch-scoped, so neither consults the override at all -- a call naming
    ///    such a context reports success but has no effect.
    ///
    /// @var replay_options
    /// @brief [CONFIG] Optional tool-provided options for this dispatch; NULL keeps the SDK's
    /// defaults. Set it during CONFIG @ref ROCPROFILER_CALLBACK_PHASE_ENTER. The SDK reads the
    /// options after that callback returns, so they must stay valid until CONFIG
    /// @ref ROCPROFILER_CALLBACK_PHASE_EXIT for the dispatch; static storage works. If the SDK
    /// cannot honor them, because @c size does not cover @c flags, a flag is unknown, or a flag
    /// names memory the SDK cannot keep a snapshot in yet, the dispatch is not replayed. An older
    /// SDK's record ends before this member, so write it only when @c size covers it.
} rocprofiler_callback_tracing_kernel_replay_data_t;

/** @} */

ROCPROFILER_EXTERN_C_FINI
