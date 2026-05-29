// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file InstrEvent.h
/// @brief Decoded, model-side view of one instruction queued for a wavefront.
///
/// The adapter builds an InstrEvent from a rocjitsu::Instruction at hook time
/// (deriving register read/write sets and pipe kind) and pushes a PendingEvent
/// into the wave's pending FIFO. Everything here is pipeline-domain — none of it
/// references rocjitsu types so the standalone cycle-model lib stays decoupled.

#pragma once

#include "cycle_model/WaitCounters.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cycle_model {

using Reg  = uint16_t;   // unified register index (VGPR space, or SGPR space)
using Tick = uint64_t;   // simdojo engine time (picoseconds); mirrors simdojo::Tick.

/// Which pipe an instruction targets. Drives ArchModel::pick_pipe routing.
enum class InstrKind : uint8_t {
  VALU, SALU, SMEM, VMEM, FLAT, GLOBAL, LDS, MFMA, WMMA, BRANCH, OTHER,
};

/// Register read/write footprint of an instruction.
///
/// Only reads matter for the RAW scoreboard (AMDGPU has no HW WAW interlock —
/// writes feed mark_writes for downstream readers only). Writes are still
/// recorded so mark_writes can set future free_cyc for RAW consumers.
struct InstrRegSet {
  std::vector<Reg> vgprs_read;
  std::vector<Reg> sgprs_read;
  std::vector<Reg> vgprs_written;
  std::vector<Reg> sgprs_written;
  bool reads_vcc = false, reads_exec = false, reads_scc = false, reads_m0 = false;
  bool writes_vcc = false, writes_exec = false, writes_scc = false, writes_m0 = false;
};

/// Real per-lane memory access shape, packed by the adapter from the functional
/// sim's computed addresses (VectorMemState / ScalarMemState). Fixed array — no
/// heap on the hot path; wave_size <= 64. The cycle-model lib coalesces these into
/// cache-line transactions (Coalescer.h) per cache level, so no rocjitsu/cache
/// geometry leaks into the adapter.
struct MemAccess {
  std::array<uint64_t, 64> lane_addr{};   // byte address per active lane
  uint64_t lane_mask = 0;                 // active-lane bitmap
  uint32_t elem_bytes = 0;                // bytes touched per active lane (elem_size*num_elems)
  // rocjitsu Mtype: UC=0/CC=1/RW=2/WB=3/NT=4. Defaults to RW (normal cached) — NOT 0,
  // because 0 is UC (bypass all caches); an unset/unfilled access must model a cached
  // access, not a bypass. The adapter overwrites this from the real Mtype; UC is only
  // ever set explicitly.
  uint8_t  mtype = 2;                      // RW
  bool     non_temporal = false;          // skip-L1 hint
};

/// A normal instruction event in the pending FIFO.
struct InstrEvent {
  std::string  mnemonic;                 // rocjitsu Instruction::mnemonic(); latency map key
  InstrKind    kind = InstrKind::OTHER;
  InstrRegSet  regs;

  // Memory-only fields (read only when kind in {SMEM,VMEM,FLAT,GLOBAL,LDS}).
  WaitCounter  wcnt = WaitCounter::VMCNT;   // counter slot this memop increments
  // Per-lane address stream — heap-allocated ONLY for memory instructions. MemAccess is
  // 528 bytes (the 64-entry lane_addr array); inlining it here would force every event
  // for the ~90% non-memory instruction stream to zero-init + memcpy 528 bytes on the
  // per-instruction ingest hot path. A null pointer for non-memory ops keeps InstrEvent
  // small (move-only). Allocated by the adapter's make_pending_event for memory kinds;
  // only ever dereferenced on is_memory()-guarded paths, where it is non-null.
  std::unique_ptr<MemAccess> mem;
};

/// Positional waitcnt gate: blocks the wave only once it reaches the FIFO head
/// and the current outstanding counts still exceed the targets. Modeled as a
/// queue entry — never a wave-global flag — so a later s_waitcnt cannot block
/// instructions queued before it.
struct WaitcntGate {
  WaitTarget target;   // per-slot thresholds; satisfied(outstanding) clears the gate
};

/// Positional barrier gate: blocks at FIFO head until its workgroup's barrier
/// resolves (signalled by the adapter from onAmdgpuBarrierResolved).
struct BarrierGate {
  uint32_t wg_id = 0;
};

enum class PendingKind : uint8_t { Instruction, WaitcntGate, BarrierGate };

/// One slot in a wave's pending FIFO. Exactly one of the union-like members is
/// meaningful per `kind`. `issued` marks a gate whose issue slot has been
/// consumed but which still blocks (waiting on counters / barrier resolution).
struct PendingEvent {
  PendingKind kind = PendingKind::Instruction;
  bool        metadata_ready = false;  // false until the adapter patches decoded fields
  bool        issued = false;          // true once a gate consumed its issue slot

  InstrEvent  instr;     // kind == Instruction
  WaitcntGate waitcnt;   // kind == WaitcntGate
  BarrierGate barrier;   // kind == BarrierGate
};

/// Shared pipe-latency descriptor — loaded from UarchConfig (per-pipe defaults and
/// the per-opcode latency table), consumed by the scheduler.
struct PipeSpec {
  uint32_t issue_rate = 1;     // cycles between successive issues on this pipe
  uint32_t base_latency = 1;   // pipeline depth (cycles to result availability)
};

}  // namespace cycle_model
