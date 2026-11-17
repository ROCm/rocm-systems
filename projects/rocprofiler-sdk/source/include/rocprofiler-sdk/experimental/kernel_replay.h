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
 * @brief A kernel dispatch as a kernel replay pass submits it.
 *
 * Sizes are in work-items, except @c cluster_size, which counts workgroups per cluster. A
 * @c cluster_size of {0, 0, 0} means an ordinary kernel dispatch packet; a nonzero one means an
 * AMD extended kernel dispatch packet (gfx1250), whose grid is a whole number of clusters. HIP on
 * gfx1250 submits ordinary launches as extended packets with a cluster size of {1, 1, 1}.
 *
 * The tool fills this struct for @c replay_set_dispatch, so @c size must be set to the size the
 * tool was compiled with. Members are only ever appended, and an SDK accepts any size that covers
 * the members it requires.
 */
typedef struct rocprofiler_kernel_replay_dispatch_t
{
    uint64_t           size;                  ///< Size of this struct
    uint64_t           kernel_object;         ///< Kernel descriptor address
    void*              kernarg_address;       ///< Kernel argument segment
    rocprofiler_dim3_t workgroup_size;        ///< Work-items per workgroup
    rocprofiler_dim3_t grid_size;             ///< Work-items in the grid
    rocprofiler_dim3_t cluster_size;          ///< Workgroups per cluster, or {0, 0, 0}
    uint32_t           group_segment_size;    ///< Group (LDS) bytes per workgroup
    uint32_t           private_segment_size;  ///< Private (scratch) bytes per work-item
} rocprofiler_kernel_replay_dispatch_t;

/**
 * @brief SDK-provided callback replacing the dispatch the current replay pass submits.
 */
typedef rocprofiler_status_t (*rocprofiler_kernel_replay_dispatch_cb_t)(
    const rocprofiler_kernel_replay_dispatch_t* dispatch);

/**
 * @brief ROCProfiler Kernel Replay Callback Tracer Record.
 *
 * Payload for @ref ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY callbacks.
 * All members are present in the struct (no unions). Which members are meaningful
 * depends on the current operation:
 *
 * - @ref ROCPROFILER_KERNEL_REPLAY_CONFIG: @c dispatch_info is populated by the SDK.
 *   The tool sets @c replay_pass_count and optionally @c replay_continue during
 *   @ref ROCPROFILER_CALLBACK_PHASE_ENTER. Pass-info fields are zero.
 * - @ref ROCPROFILER_KERNEL_REPLAY_PASS: @c dispatch_info, @c current_pass, and
 *   @c total_passes are populated by the SDK. @c replay_pass_count reads as NULL and must not be
 *   modified (the pass count is fixed at CONFIG PHASE_ENTER). @c replay_continue is NULL during
 *   PASS PHASE_ENTER; during PASS PHASE_EXIT the SDK populates it with the callback in effect for
 *   this pass, and the tool may overwrite it there to change the continue decision for this pass
 *   only (the next pass reverts to the CONFIG default).
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
 * variables. Unified or managed memory, @c hipMallocAsync and other virtual-memory-mapped
 * allocations, and host, fine-grained and kernarg memory are not captured, so a kernel writing to
 * them observes values accumulated across passes rather than identical inputs. Allocations carrying
 * @c HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG are excluded from the snapshot (HIP kernarg pools /
 * profiler buffers share that flag). A direct-HSA application that puts ordinary writable device
 * data behind the same flag sees the same omission -- an unsupported allocation class for beta.
 * Declining every replay while any such trackable allocation is live is not viable because the HIP
 * runtime itself keeps them live under interception. A failed restore aborts the process rather
 * than continuing with partially restored memory.
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
    const rocprofiler_kernel_replay_dispatch_t* app_dispatch;
    rocprofiler_kernel_replay_dispatch_cb_t     replay_set_dispatch;

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
    /// @var app_dispatch
    /// @brief [CONFIG/PASS] The dispatch the application submitted, read-only, populated by the
    /// SDK. Valid only for the duration of the callback; copy what is needed. Its
    /// @c kernarg_address points at the application's kernel arguments, which stay unchanged
    /// until the replay loop ends.
    ///
    /// @var replay_set_dispatch
    /// @brief [PASS] Replace the dispatch the current pass submits. The SDK populates this
    /// function pointer before each PASS @ref ROCPROFILER_CALLBACK_PHASE_ENTER. Semantics:
    ///  - Only valid to call during PASS @ref ROCPROFILER_CALLBACK_PHASE_ENTER; otherwise it
    ///    returns @ref ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR.
    ///  - Applies to the current pass only. A pass that does not call it submits the
    ///    application's dispatch; @c NULL clears a replacement made earlier in the same callback.
    ///  - The SDK copies @p dispatch. The memory at @c kernarg_address must stay valid and
    ///    unchanged until the pass's @ref ROCPROFILER_CALLBACK_PHASE_EXIT.
    ///  - The packet keeps the application's packet type, header and fences: an ordinary
    ///    dispatch packet needs @c cluster_size {0, 0, 0}, and an extended dispatch packet needs a
    ///    nonzero @c cluster_size with a grid that is a whole number of clusters. Start from
    ///    @c app_dispatch to keep both right.
    ///  - @c kernel_object must belong to a code object loaded through HSA on the replayed agent,
    ///    so the SDK knows its kernel id. Sizes must be nonzero and fit the packet's fields.
    ///    @c kernarg_address may be NULL only for a kernel without kernel arguments. A rejected
    ///    call returns @ref ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT and keeps the previous
    ///    choice.
    ///  - Everything else replay does is unchanged: the memory snapshot and restore, the
    ///    profiling packets injected around the kernel, the dispatch id, and the deferred
    ///    application completion signal. Services observing the pass (dispatch tracing, counters,
    ///    thread trace) see the replacement's kernel id and geometry; @c dispatch_info in this
    ///    payload keeps describing the application's dispatch.
    ///  - The application keeps the output of the last pass that runs, so a tool replacing
    ///    dispatches normally lets the final pass submit the application's dispatch.
    ///  - A tool must check that @c size covers this member before calling it.
} rocprofiler_callback_tracing_kernel_replay_data_t;

/** @} */

ROCPROFILER_EXTERN_C_FINI
