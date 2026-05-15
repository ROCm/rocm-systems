////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

// HSA-resident firmware-ring drainer with LTTng emission for kernel-dispatch
// timestamps. Public API surface for the dispatch_log subsystem. See
// projects/rocr-runtime/runtime/hsa-runtime/core/runtime/dispatch_log.cpp
// for the implementation, and the Phase A spec (2026-04-27) for the full
// design contract.
//
// Draining is signal-bound: AqlQueue::GetDispatchLogPointers supplies host-VA
// wptr/rptr/signal counters (KFD_IOC_PROFILER_DISPATCH_LOG). SetProfiling(true)
// fails if that registration cannot complete; there is no sentinel-scan path.

#ifndef HSA_RUNTME_CORE_INC_DISPATCH_LOG_H_
#define HSA_RUNTME_CORE_INC_DISPATCH_LOG_H_

#include <cstddef>
#include <cstdint>

#include "hsa.h"

namespace rocr {

namespace core {
class Queue;
}  // namespace core

namespace dispatch_log {

// -----------------------------------------------------------------------------
// On-wire / in-buffer constants. See spec §5 (ring layout) and §6 (record
// format). The 16-byte record format is the MEC firmware contract.
//
// The buffer layout was originally defined as `header + ring data`. The
// current substrate instead hands the drainer a flat ring of N records
// at the 16-byte FW record stride. The drainer is signal-bound: it reads
// *signal_ptr from AqlQueue::GetDispatchLogPointers and walks [next_idx,
// observed), using record_type==0 inside that window for multi-XCC /
// init-sync quarantine (see drain_one_queue). The original buffer header
// and DISPATCH_LOG_VERSION are no longer used; they are retained here only
// as historical constants for now and may be removed.
// -----------------------------------------------------------------------------

// NOTE: the per-queue ring record count is NOT a build-time constant.
// AqlQueue::SetProfiling chooses the num_records value at allocation
// time and the drainer derives ring_records at runtime from
// AqlQueue::GetProfilingDispatchRecords (buf_bytes / sizeof(record)).
// A previous DISPATCH_LOG_RECORD_COUNT constant was removed because it
// silently went stale when SetProfiling's ring size was bumped (and was
// not actually used by the drainer code path). The single source of
// truth is AqlQueue::SetProfiling in core/runtime/amd_aql_queue.cpp;
// consumers MUST query GetProfilingDispatchRecords for the live size.

// FW-written 16-byte record. Layout is fixed by the firmware contract.
// Mirrors rocr::AMD::mec_dispatch_record from
// core/inc/mec_dispatch_record.h, with the trailing 4-byte field treated
// as the dispatch index (FW writes the per-queue monotonic dispatch idx
// into the slot named `reserved` in mec_dispatch_record; see that header
// for the historical-name vs current-FW-semantics rationale).
//
// `record_type` is FW-defined and intentionally left opaque to HSA: the
// drainer does not interpret which value means dispatch start vs end and
// emits one rocm_hsa:kernel_dispatch_record event per non-zero record
// regardless of the tag value. Stream consumers that want paired
// start/end semantics join records on (queue_id, dispatch_idx) and
// either order by gpu_ts or apply their own FW-version-specific
// record_type interpretation. record_type==0 marks an empty slot (buffer
// pre-zeroed; also used inside the signal-bound scan for quarantine).
#pragma pack(push, 1)
struct mec_dispatch_record_16 {
  uint32_t ts_lo;          // GPU clock low 32 bits
  uint32_t ts_hi;          // GPU clock high 32 bits
  uint32_t record_type;    // FW-defined; 0 marks empty / quarantine-relevant slot
  uint32_t dispatch_idx;   // FW-written dispatch idx (mec_dispatch_record::reserved)
};
#pragma pack(pop)
static_assert(sizeof(mec_dispatch_record_16) == 16,
              "mec_dispatch_record_16 must be exactly 16 bytes (FW contract)");

// -----------------------------------------------------------------------------
// Lifecycle hooks. Called from Runtime::Load / Runtime::Unload and from the
// hsa_queue_create / hsa_queue_destroy entry points. See spec §8 file layout
// and §4 lifecycle state machine.
// -----------------------------------------------------------------------------

// Called once from Runtime::Load. Probes for KFD substrate support and spawns
// the ts_poller thread. Per-queue drainer worker threads are spawned by the
// poller (or by on_queue_create) on each per-queue enable edge — one worker
// per active queue, joined on its disable edge (spec §7).
void init();

// Called once from Runtime::Unload. Joins the poller, then runs the disable
// edge for every active queue, which stops + joins each per-queue drainer
// worker.
void shutdown();

// Called immediately AFTER a queue becomes visible via hsa_queue_create. If
// the tracepoint is currently enabled, runs the per-queue ENABLE sequence
// from spec §5 on the calling thread (bounded latency: one host alloc + one
// KFD ioctl + one MQD-flush round-trip).
void on_queue_create(core::Queue* q);

// Called BEFORE a queue is destroyed via hsa_queue_destroy. If the queue has
// dispatch_log_active=true, runs the per-queue DISABLE sequence from spec §4
// queue-destroy hook on the calling thread (bounded final drain + KFD ioctl).
void on_queue_destroy(core::Queue* q);

// -----------------------------------------------------------------------------
// Profiling-bit refcount API (spec §4a). Multiple consumers (LTTng dispatch-log
// path, rocprofiler-sdk dispatch-timestamp path, future profilers) must
// coordinate through this refcount instead of writing
// AMD_QUEUE_PROPERTIES_ENABLE_PROFILING (bit 3 in amd_queue_t.queue_properties)
// directly. The bit is set whenever refcount > 0 and clear whenever
// refcount == 0; every Acquire MUST be matched by exactly one Release.
//
// rocprofiler-sdk migration to this API is a hard precondition for shipping
// the LTTng dispatch-log feature (spec §10 dep #4). Phase A (this scaffolding)
// can land independently of the SDK migration; the LTTng path uses these
// helpers unconditionally.
// -----------------------------------------------------------------------------

hsa_status_t QueueProfilingAcquire(core::Queue* q);
hsa_status_t QueueProfilingRelease(core::Queue* q);

}  // namespace dispatch_log
}  // namespace rocr

#endif  // HSA_RUNTME_CORE_INC_DISPATCH_LOG_H_
