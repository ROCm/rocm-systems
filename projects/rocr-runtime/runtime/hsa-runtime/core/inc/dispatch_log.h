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
// format). The 16-byte record format and 64-byte header layout are frozen at
// version 1; future revisions bump DISPATCH_LOG_VERSION and the drainer
// refuses to consume rings whose header version it does not recognize.
// -----------------------------------------------------------------------------

// Ring header is one cache line preceding the FW-written data area.
constexpr size_t  DISPATCH_LOG_HEADER_BYTES = 64;

// Header / record format version stamped into the ring header at ENABLE
// and checked by the drainer on first access. Bump on any change to the
// 16-byte record layout or the 64-byte header layout.
constexpr uint32_t DISPATCH_LOG_VERSION = 1;

// Power-of-2 size of the ring data area (NOT including the 64-byte header).
// 4 MB matches the spec's typical-budget recommendation in §5.
constexpr size_t  DISPATCH_LOG_RING_BYTES = 4 * 1024 * 1024;

// Record-type tags written by FW into mec_dispatch_record_16::record_type.
enum {
  DISPATCH_LOG_RECORD_START = 1,
  DISPATCH_LOG_RECORD_END   = 2,
};

// FW-written 16-byte record. Layout is fixed by the firmware contract: see
// spec §6 for the field semantics and §14 glossary for the dispatch_idx
// 32-bit width contract. Packed to 16 bytes with no implicit padding.
#pragma pack(push, 1)
struct mec_dispatch_record_16 {
  uint32_t ts_lo;          // GPU clock low 32 bits
  uint32_t ts_hi;          // GPU clock high 32 bits
  uint32_t record_type;    // DISPATCH_LOG_RECORD_{START,END}
  uint32_t dispatch_idx;   // low 32 bits of AQL read_dispatch_id
};
#pragma pack(pop)
static_assert(sizeof(mec_dispatch_record_16) == 16,
              "mec_dispatch_record_16 must be exactly 16 bytes (FW contract)");

// Ring header layout (B + 0 .. B + DISPATCH_LOG_HEADER_BYTES). FW writes
// `fw_wptr`; HSA writes `generation` and `version` at ENABLE. See spec §5.
struct dispatch_log_header {
  volatile uint64_t fw_wptr;     // monotonic byte counter, FW-written
  uint64_t          generation;  // bumped by HSA on each per-queue ENABLE
  uint32_t          version;     // header/record-format version
  uint32_t          reserved[11];
};
static_assert(sizeof(dispatch_log_header) == DISPATCH_LOG_HEADER_BYTES,
              "dispatch_log_header must occupy exactly one cache line "
              "(DISPATCH_LOG_HEADER_BYTES)");

// -----------------------------------------------------------------------------
// Lifecycle hooks. Called from Runtime::Load / Runtime::Unload and from the
// hsa_queue_create / hsa_queue_destroy entry points. See spec §8 file layout
// and §4 lifecycle state machine.
// -----------------------------------------------------------------------------

// Called once from Runtime::Load. Probes for KFD substrate support and spawns
// the ts_poller thread. The drainer thread is spawned lazily by ts_poller on
// the first enable edge (spec §9 LTTng-OFF row).
void init();

// Called once from Runtime::Unload. Runs the disable edge for every active
// queue, then joins the poller and (if running) drainer threads.
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
