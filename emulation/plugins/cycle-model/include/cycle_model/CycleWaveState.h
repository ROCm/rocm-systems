// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file CycleWaveState.h
/// @brief Per-wavefront cycle-model state. Exactly one instance per resident
/// wavefront. In a fully-extended PR #6132 this would hang off a Wavefront
/// plugin-state slot; today the adapter owns it in a side map keyed on the
/// wavefront's (cu, wf_id, dispatch_id) identity.
///
/// "Per-wavefront" — distinct from SIMDUnitState (one per HW SIMD unit, shared by
/// the ~8-10 waves on that SIMD) and CUState (one per compute unit).

#pragma once

#include "cycle_model/InstrEvent.h"
#include "cycle_model/MemReqStateMachine.h"
#include "cycle_model/WaveScoreboard.h"

#include <cstdint>
#include <deque>
#include <unordered_map>

namespace cycle_model {

struct CycleWaveState {
  // Implicit-register RAW readiness (reads-only scoreboard).
  Scoreboard scoreboard;

  // Pending FIFO — program-order queue of not-yet-modeled events. Instruction,
  // waitcnt-gate, and barrier-gate entries are all positional here.
  std::deque<PendingEvent> pending;

  // Wave-local readiness — FINITE cycle, never UINT64_MAX. The earliest CU cycle
  // this wave could issue, expressed in the shared cu_cycle axis.
  uint64_t next_ready_cyc = 0;

  // Permanent SIMD assignment (set at dispatch). Indexes ArchModel::simds.
  uint32_t simd_id = 0;

  // Cycle-domain wait counters (independent of funcsim's wait_counters, which
  // drain synchronously). Incremented at memop commit, decremented at retire.
  WaitCounters outstanding;

  // Set true by the adapter on onAmdgpuBarrierResolved; consumed (reset) when the
  // head BarrierGate clears. 1:1 with barrier-resolve events in program order.
  bool barrier_signaled = false;

  // Model-owned in-flight memory requests + tombstones for dup/late detection.
  std::unordered_map<MemReqId, MemReqEntry> in_flight;
  TombstoneSet completed_rids;
  TombstoneSet halted_rids;

  // Observational lifecycle counters (never alter modeled cycle counts).
  MemReqCounters mem_counters;

  uint64_t cyc_at_dispatch = 0;
  uint64_t cyc_at_halt = 0;

  /// Cleared at dispatch (slot reuse). Tombstones + mem_counters are intentionally
  /// NOT cleared — monotonic MemReqId means no cross-dispatch collision, and the
  /// counters are cumulative observational metrics.
  void reset_for_dispatch(uint64_t cu_cycle) {
    scoreboard.clear();
    pending.clear();
    in_flight.clear();
    outstanding.clear();
    barrier_signaled = false;
    next_ready_cyc = 0;
    cyc_at_dispatch = cu_cycle;
  }
};

}  // namespace cycle_model
