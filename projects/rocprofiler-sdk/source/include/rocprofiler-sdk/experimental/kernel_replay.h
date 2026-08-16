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
 * @brief ROCProfiler Kernel Replay Callback Tracer Record.
 *
 * Payload for @ref ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY callbacks.
 * All members are present in the struct (no unions). Which members are meaningful
 * depends on the current operation:
 *
 * - @ref ROCPROFILER_KERNEL_REPLAY_CONFIG: @c dispatch_info is populated by the SDK.
 *   The tool sets @c pass_count_cb and optionally @c replay_continue_cb during
 *   @ref ROCPROFILER_CALLBACK_PHASE_ENTER. Pass-info fields are zero.
 * - @ref ROCPROFILER_KERNEL_REPLAY_PASS: @c dispatch_info, @c current_pass, and
 *   @c total_passes are populated by the SDK. Config fields are zero/null and must not be
 *   modified.
 *
 * The SDK maintains a single @c rocprofiler_user_data_t for the entire replay sequence
 * (CONFIG + all PASS operations). A tool can write per-dispatch state into
 * @c user_data during CONFIG PHASE_ENTER; the same value is delivered to every
 * subsequent PASS callback and to @c pass_count_cb and @c replay_continue_cb for the
 * same dispatch.
 *
 * CONFIG and PASS callbacks run on the dispatch's launch thread inside the HSA write
 * interceptor. Tools must not call @ref rocprofiler_start_context /
 * @ref rocprofiler_stop_context from those callbacks: the interceptor may already be
 * draining services, and a global start/stop can deadlock or observe a half-updated
 * context list. Use the localized start/stop pointers on this payload instead.
 *
 * @warning Kernel replay is experimental. Snapshot/restore covers the agent's tracked
 * coarse-grained HSA pool/region allocations plus loaded-module variables
 * (@c HSA_SYMBOL_KIND_VARIABLE). It does **not** cover unified/managed memory, the
 * HSA virtual-memory path (@c hsa_amd_vmem_map / @c hipMallocAsync), or allocations
 * created with @c HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG. Replay is single AQL packet
 * only; HIP graph launches are not supported. Host caches, atomics, host RAM, and the
 * HBM envelope around a restore are outside the snapshot contract -- a kernel that
 * depends on those for pass-to-pass identity is not a valid replay candidate.
 *
 * Only one process-wide @ref ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY subscriber is
 * permitted. A second configure returns
 * @ref ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED.
 */
typedef struct rocprofiler_callback_tracing_kernel_replay_data_t
{
    uint64_t                           size;           ///< sizeof this struct (versioning)
    rocprofiler_kernel_dispatch_info_t dispatch_info;  ///< Kernel dispatch info (always set)

    /// @brief [CONFIG] Tool-provided callback returning the number of replay passes.
    /// The tool sets this during CONFIG @ref ROCPROFILER_CALLBACK_PHASE_ENTER; the SDK
    /// then calls it (if non-null) to obtain the pass count for this dispatch:
    ///  - left NULL     => dispatch is NOT replayed; it runs once and execution
    ///                     continues as usual (no snapshot, per-dispatch opt-out)
    ///  - returns N > 1 => fixed loop of N passes (optional @c replay_continue_cb may
    ///                     break early; it is not invoked after the final fixed pass)
    ///  - returns 1     => not replayed (one execution, no snapshot), even if
    ///                     @c replay_continue_cb is set
    ///  - returns 0     => indefinite loop if @c replay_continue_cb is set; if it is
    ///                     not, the SDK logs a warning and runs the dispatch once
    /// CONFIG PHASE_EXIT fires on every decline (NULL / N==0 without continue / N==1)
    /// and after a real replay loop.
    /// @c dispatch_info and @c user_data are provided so the tool can pick the count
    /// per dispatch and thread per-dispatch state through the callbacks.
    uint64_t (*pass_count_cb)(rocprofiler_kernel_dispatch_info_t dispatch_info,
                              rocprofiler_user_data_t            user_data);

    /// @brief [CONFIG] Optional tool-provided callback invoked after each pass completes.
    /// Return non-zero to continue the replay loop, zero to break out.
    /// Required when @c pass_count_cb returns 0; if it returns N > 0, allows early exit.
    /// @c dispatch_info and @c user_data (the per-dispatch user data set during CONFIG
    /// PHASE_ENTER) are provided; the same @c user_data is threaded through all callbacks
    /// for this dispatch.
    int (*replay_continue_cb)(rocprofiler_kernel_dispatch_info_t dispatch_info,
                              uint64_t                           current_pass,
                              uint64_t                           total_passes,
                              rocprofiler_user_data_t            user_data);

    /// @brief [PASS] 0-indexed current pass number. Read-only, populated by SDK.
    uint64_t current_pass;

    /// @brief [PASS] Total passes if known (the value returned by @c pass_count_cb),
    /// else 0 for an indefinite loop. Read-only.
    uint64_t total_passes;

    /// @brief [PASS] Localized context control. The SDK populates these function
    /// pointers before each PASS @ref ROCPROFILER_CALLBACK_PHASE_ENTER; the tool
    /// invokes them (in lieu of the global @ref rocprofiler_start_context /
    /// @ref rocprofiler_stop_context) to enable or disable a context for the current
    /// replay loop. Semantics:
    ///  - Only valid to call during PASS @ref ROCPROFILER_CALLBACK_PHASE_ENTER.
    ///  - Sticky across passes: a context stopped in one pass stays stopped until it
    ///    is started again within the same replay loop (and vice versa).
    ///  - Scoped to the replay loop: each context's pre-replay active/inactive state
    ///    is restored once the loop completes. Global context state is never modified.
    ///  - Honored for kernel-dispatch tracing, dispatch counter collection, and
    ///    dispatch thread trace. PC sampling, SPM, and device-wide counters ignore
    ///    the override (no-op); starting a context that has only those services
    ///    returns @ref ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED.
    ///  - Cannot promote a globally inactive context: local start of a registered
    ///    context that is not in the active set returns
    ///    @ref ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED. Local stop of an active
    ///    context is the supported direction.
    rocprofiler_status_t (*replay_local_start_context_cb)(rocprofiler_context_id_t context_id);
    rocprofiler_status_t (*replay_local_stop_context_cb)(rocprofiler_context_id_t context_id);
} rocprofiler_callback_tracing_kernel_replay_data_t;

/** @} */

ROCPROFILER_EXTERN_C_FINI
