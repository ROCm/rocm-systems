// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file CuState.h
/// @brief Pipe and CU/SIMD timing state. Three ownership tiers:
///   CycleWaveState  — one per wavefront (elsewhere).
///   SIMDUnitState   — one per HW SIMD unit; owns VALU/MFMA pipes + front-end slot.
///   CUState         — one per compute unit; owns the shared scalar/memory pipes,
///                     the cu_cycle clock, and the monotonic MemReqId allocator.
///
/// The per-SIMD array of SIMDUnitState lives on ArchModel (runtime-sized from
/// UarchConfig::simds_per_cu), NOT on CUState — CUState only carries genuinely
/// CU-wide state.

#pragma once

#include "cycle_model/CycleWaveState.h"
#include "cycle_model/InstrEvent.h"

#include <cstdint>
#include <vector>

namespace cycle_model {

/// A single pipe's structural-hazard tracker. Plan text uses "Pipe" and
/// "IssueQueue" interchangeably; this is the one canonical type.
struct IssueQueue {
  uint64_t next_free_cyc = 0;   // earliest cycle this pipe can accept a new issue
  uint32_t issue_rate = 1;      // cycles a single issue occupies the pipe
};
using Pipe = IssueQueue;

/// CU-wide aggregate stall accounting (observational counters only).
struct StallCounters {
  uint64_t sched = 0;        // CU cycles where nothing issued and no terminal pending
  uint64_t pipe_busy = 0;    // CU cycles a wave was ready but its pipe was occupied
};

/// One hardware SIMD unit. Pipes and the front-end issue slot here are SHARED by
/// every resident wavefront assigned to this SIMD (see CycleWaveState::simd_id).
struct SIMDUnitState {
  IssueQueue valu;            // per-SIMD vector ALU
  IssueQueue mfma;            // per-SIMD matrix pipe (unused/idle on archs lacking MFMA)
  IssueQueue wmma;            // per-SIMD WMMA pipe (unused on CDNA)

  // Per-SIMD front-end issue port — caps issues at front_end_issue_per_simd per cycle.
  uint64_t front_end_accounting_cyc = 0;
  uint32_t front_end_issues_used = 0;
  uint64_t front_end_blocked_until_cyc = 0;

  // Wavefronts the dispatcher assigned to this SIMD.
  std::vector<CycleWaveState*> waves;
  uint32_t next_wave_idx = 0;   // round-robin cursor within this SIMD
};

/// One compute unit. Holds the shared clock, CU-wide shared pipes, scheduler
/// rotation across SIMDs, and the monotonic memory-request id allocator.
struct CUState {
  uint64_t cu_cycle = 0;          // the shared CU clock — the time axis everything contends in

  // CU-wide shared pipes — one physical port each; all SIMDs contend.
  IssueQueue salu;
  IssueQueue smem;
  IssueQueue vmem;                // VMEM/FLAT/global via TCP
  IssueQueue lds;                 // LDS issue port (bank arbitration folded into latency)

  StallCounters stalls;
  uint32_t next_simd_idx = 0;     // cross-cycle round-robin rotation across SIMDs

  uint64_t next_mem_rid = 0;      // monotonic MemReqId allocator; ctor-init only, never reset
  MemReqId alloc_mem_req_id() { return next_mem_rid++; }
};

}  // namespace cycle_model
