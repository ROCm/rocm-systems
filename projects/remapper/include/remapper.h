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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

/**
 * @file remapper.h
 * @brief Workgroup remapper: multi-SIMD thread trace of one kernel dispatch on
 * gfx1250.
 *
 * A profiler drives the remapper from rocprofiler-sdk kernel replay callbacks:
 *
 * - CONFIG PHASE_ENTER: @ref remapper_begin_dispatch rewrites the kernel, loads
 * it through HSA and allocates its device buffers. A NULL result means the
 * dispatch runs without replay. Otherwise the profiler asks for an indefinite
 * loop (`replay_pass_count` returns 0) whose `replay_continue` reports the last
 * @ref remapper_pass_exit result.
 * - PASS PHASE_ENTER: @ref remapper_pass_enter fills a @ref
 * remapper_pass_plan_t. When
 *   @c replace_dispatch is set, the profiler passes @c dispatch to
 * `replay_set_dispatch`; when
 *   @c trace is set, it traces the pass with the given target, and otherwise
 * not at all.
 * - PASS PHASE_EXIT: @ref remapper_pass_exit reads the pass's logs (before the
 * SDK restores device memory) and returns whether another pass is needed. The
 * last pass always submits the application's own dispatch, untraced.
 * - CONFIG PHASE_EXIT: @ref remapper_end_dispatch frees the session.
 *
 * The remapper never submits work to the application's queues: device buffers
 * are written and read with @c hsa_memory_copy. Each remapped dispatch writes
 * `<output_dir>/dispatch-<id>/` (rewritten code object, manifest, report and
 * the pass list the ATT merge reads).
 *
 * Every function is safe to call from any thread; a session is used by one
 * thread at a time.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Incremented when a struct or function changes incompatibly. */
#define REMAPPER_ABI_VERSION 1

typedef struct remapper_t remapper_t;
typedef struct remapper_session_t remapper_session_t;

typedef struct remapper_dim3_t {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} remapper_dim3_t;

/** A kernel dispatch, as in `rocprofiler_kernel_replay_dispatch_t`. */
typedef struct remapper_dispatch_t {
  uint64_t kernel_object;         ///< Kernel descriptor address
  uint64_t kernarg_address;       ///< Kernel argument segment
  remapper_dim3_t workgroup_size; ///< Work-items per workgroup
  remapper_dim3_t grid_size;      ///< Work-items in the grid
  remapper_dim3_t cluster_size;   ///< Workgroups per cluster, or {0, 0, 0}
  uint32_t group_segment_size;    ///< Group (LDS) bytes per workgroup
  uint32_t private_segment_size;  ///< Private (scratch) bytes per work-item
} remapper_dispatch_t;

/** The dispatch a remapper session is asked to take over. */
typedef struct remapper_kernel_t {
  uint64_t size; ///< sizeof(remapper_kernel_t)
  uint64_t dispatch_id;
  const char *kernel_name; ///< Kernel symbol, with or without ".kd"
  const void *code_object; ///< The loaded code object (an ELF for the agent)
  uint64_t code_object_size;
  uint32_t agent_node_id; ///< KFD node id of the agent
                          ///< (rocprofiler_agent_t::node_id)
  uint32_t reserved;
  remapper_dispatch_t dispatch; ///< The application's dispatch
} remapper_kernel_t;

/** What one replay pass submits and traces. */
typedef struct remapper_pass_plan_t {
  uint64_t size; ///< Set by the caller to sizeof(remapper_pass_plan_t)
  int32_t
      replace_dispatch; ///< Nonzero: submit @c dispatch instead of the app's
  int32_t trace;        ///< Nonzero: thread trace this pass
  remapper_dispatch_t dispatch;
  uint32_t target_cu; ///< Thread trace target CU (WGP)
  uint32_t simd;      ///< Thread trace SIMD
  uint64_t shader_engine_mask;
} remapper_pass_plan_t;

/**
 * Create a remapper writing under @p output_dir.
 *
 * @p options is NULL or a comma-separated `key=value` list: records, checks,
 * drift-retries, repetitions, target-cu, wgps-per-se, launch-model (auto,
 * table, sparse, admission), prefetch (per-wave, wave0, none), natural-baseline
 * (0, 1), original-baseline (0, 1), admission-slack, loop-markers (cluster,
 * cluster-and-workgroup), shader-engines, max-dispatches, table-capacity-bytes.
 *
 * @return NULL on error; see @ref remapper_last_error.
 */
remapper_t *remapper_create(const char *output_dir, const char *options);

void remapper_destroy(remapper_t *remapper);

/** Nonzero while the remapper would take another dispatch. */
int remapper_wants_dispatch(remapper_t *remapper);

/**
 * Prepare a dispatch for remapping. Call before the replay loop starts.
 *
 * @return NULL when the dispatch should run without replay (the reason is in
 * @ref remapper_last_error and in the dispatch's report.json).
 */
remapper_session_t *remapper_begin_dispatch(remapper_t *remapper,
                                            const remapper_kernel_t *kernel);

/**
 * Plan pass @p pass and write its control block and table.
 *
 * @return 0 on success. On error @p plan asks for the application's dispatch,
 * untraced.
 */
int remapper_pass_enter(remapper_session_t *session, uint64_t pass,
                        remapper_pass_plan_t *plan);

/**
 * Finish pass @p pass: read its logs and advance the schedule.
 *
 * @return 1 if the loop needs another pass, 0 if it is done, -1 on error (stop
 * the loop).
 */
int remapper_pass_exit(remapper_session_t *session, uint64_t pass);

/** The session's output directory, valid until @ref remapper_end_dispatch. */
const char *remapper_session_dir(remapper_session_t *session);

/**
 * Free a session.
 *
 * @return 0 if every stage succeeded, 1 if the remap failed (see the report),
 * -1 if the loop ended before the schedule did.
 */
int remapper_end_dispatch(remapper_session_t *session);

/** REMAPPER_ABI_VERSION of the library; a caller that copies these declarations
 * checks it after dlopen. */
uint32_t remapper_abi_version(void);

/** The calling thread's last error message, or an empty string. */
const char *remapper_last_error(void);

#ifdef __cplusplus
}
#endif
