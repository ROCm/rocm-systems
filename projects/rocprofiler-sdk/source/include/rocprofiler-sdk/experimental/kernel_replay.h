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
 *   @c total_passes are populated by the SDK. Config fields are zero/null and must not be
 *   modified.
 *
 * The SDK maintains a single @c rocprofiler_user_data_t for the entire replay sequence
 * (CONFIG + all PASS operations). A tool can write per-dispatch state into
 * @c user_data during CONFIG PHASE_ENTER; the same value is delivered to every
 * subsequent PASS callback and to @c replay_pass_count and @c replay_continue for the
 * same dispatch.
 *
 * Exactly one context may configure a KERNEL_REPLAY service; a second
 * @ref rocprofiler_configure_callback_tracing_service for this domain returns
 * @ref ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED. @c user_data written during a PASS
 * callback is not retained -- only the value set during CONFIG @ref
 * ROCPROFILER_CALLBACK_PHASE_ENTER is threaded through the sequence.
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
    uint64_t                                  size;  ///< Size of this struct
    rocprofiler_kernel_dispatch_info_t        dispatch_info;  ///< Kernel dispatch info (always set)
    uint64_t                                  current_pass;
    uint64_t                                  total_passes;
    rocprofiler_kernel_replay_pass_count_cb_t replay_pass_count;
    rocprofiler_kernel_replay_continue_cb_t   replay_continue;
    rocprofiler_kernel_replay_context_cb_t    replay_start_context;
    rocprofiler_kernel_replay_context_cb_t    replay_stop_context;

    /// @var replay_pass_count
    /// @brief [CONFIG] Tool-provided callback returning the number of replay passes.
    /// The tool sets this during CONFIG @ref ROCPROFILER_CALLBACK_PHASE_ENTER; the SDK
    /// then calls it (if non-null) to obtain the pass count for this dispatch:
    ///  - left NULL     => dispatch is NOT replayed; it runs once and execution
    ///                     continues as usual (no snapshot, per-dispatch opt-out)
    ///  - returns N > 0 => fixed loop of N passes (optional @c replay_continue honored)
    ///  - returns 0     => indefinite loop (requires @c replay_continue)
    /// @c dispatch_info and @c user_data are provided so the tool can pick the count
    /// per dispatch and thread per-dispatch state through the callbacks.
    ///
    /// @var replay_continue
    /// @brief [CONFIG] Optional tool-provided callback invoked after each pass completes.
    /// Return non-zero to continue the replay loop, zero to break out.
    /// Required when @c replay_pass_count returns 0; if it returns N > 0, allows early exit
    /// only — it cannot extend the loop past N. @c rocprofv3 does not set this callback;
    /// adaptive or early-exit control is a custom-tool feature.
    /// @c dispatch_info and @c user_data (the per-dispatch user data set during CONFIG
    /// PHASE_ENTER) are provided; the same @c user_data is threaded through all callbacks
    /// for this dispatch.
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
} rocprofiler_callback_tracing_kernel_replay_data_t;

/** @} */

ROCPROFILER_EXTERN_C_FINI
