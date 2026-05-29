// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file ArchModel.h
/// @brief Runtime, JSON-configured cycle model for one compute unit.
///
/// One ArchModel per compute unit, constructed from a UarchConfig loaded at
/// startup. SIMD count, pipe latencies, feature flags, and the per-opcode latency
/// table all come from JSON (rides rocjitsu's existing config workflow; no
/// compile-time arch traits, no variant). All five archs use this one class.

#pragma once

#include "cycle_model/CuState.h"
#include "cycle_model/InstrEvent.h"
#include "cycle_model/MemorySystem.h"
#include "cycle_model/UarchConfig.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace cycle_model {

enum class IssueResult : uint8_t {
  ISSUED, GATE_CLEARED, BLOCKED_METADATA, BLOCKED_GATE,
  BLOCKED_SCOREBOARD, BLOCKED_PIPE, BLOCKED_FRONT_END, BLOCKED_MSHR, EMPTY,
};

struct IssueDecision {
  IssueResult result = IssueResult::EMPTY;
  uint64_t    earliest_retry = UINT64_MAX;
};

struct SchedulerStepResult {
  uint64_t earliest_retry = UINT64_MAX;
  uint32_t issues = 0;
};

class ArchModel {
 public:
  explicit ArchModel(const UarchConfig& cfg) : cfg_(cfg), mem_(cfg_), simds_(cfg.simds_per_cu) {
    cu_.salu.issue_rate = cfg.salu.issue_rate;
    cu_.smem.issue_rate = cfg.smem.issue_rate;
    cu_.vmem.issue_rate = cfg.vmem.issue_rate;
    cu_.lds.issue_rate  = cfg.lds_pipe.issue_rate;
    for (auto& s : simds_) {
      s.valu.issue_rate = cfg.valu.issue_rate;
      if (cfg.has_mfma) s.mfma.issue_rate = cfg.mfma.issue_rate;
      if (cfg.has_wmma) s.wmma.issue_rate = cfg.wmma.issue_rate;
      s.waves.reserve(cfg.wave_slots_per_simd);
    }
  }

  PipeSpec latency(std::string_view mnemonic) const {
    return cfg_.lookup_latency(mnemonic, cfg_.valu);
  }

  IssueQueue& pick_pipe(InstrKind k, SIMDUnitState& simd) {
    switch (k) {
      case InstrKind::VALU: return simd.valu;
      case InstrKind::MFMA: return simd.mfma;   // unused/idle on archs without MFMA
      case InstrKind::WMMA: return simd.wmma;
      case InstrKind::SALU:                   return cu_.salu;
      case InstrKind::SMEM:                   return cu_.smem;
      case InstrKind::VMEM:
      case InstrKind::FLAT:
      case InstrKind::GLOBAL:                 return cu_.vmem;
      case InstrKind::LDS:                    return cu_.lds;
      default:                                return cu_.salu;
    }
  }

  IssueDecision       try_issue_head_at(CycleWaveState& ws, uint64_t now);
  // Fast path: issue `inst` for `ws` at `now` WITHOUT touching ws.pending.
  // Returns ISSUED (state mutated) or a BLOCKED_* reason. Lets the adapter
  // skip allocating a PendingEvent when a wave has no backlog.
  IssueResult         try_issue_inline(CycleWaveState& ws, const InstrEvent& inst, uint64_t now);
  SchedulerStepResult scheduler_step();
  // Advance exactly one CU clock cycle: retire due memory, run the scheduler,
  // step cu_cycle. Returns has_work() so the clock driver can decide whether to
  // keep ticking.
  bool tick_cycle();

  // True iff any resident wave still has a pending FIFO entry or an in-flight
  // memory request. The clock idles when this is false.
  bool has_work() const;

  // Kernel-end tail drain: advances cu_cycle until all pending FIFOs are empty.
  void                flush_to_quiescence();
  // Passive-path drain (LD_PRELOAD): the simdojo cu_clk domain is starved (the
  // functional sim never yields enough clock edges), so the engine-driven async memory
  // completion (on_mem_completion over the MemSys port) never arrives and waitcnt gates
  // would deadlock flush_to_quiescence. This drives quiescence AND synchronously
  // completes the deferred async memory still queued in mem_out_, so gates clear and the
  // whole pending backlog is issued + counted. Accuracy bound: shared L2/HBM contention
  // latency is NOT applied to these force-completed requests (they retire at cu_cycle);
  // synchronous (compute / L1-hit) latencies and intra-CU issue/stall timing are exact.
  void                drive_to_quiescence_passive();
  // Per-dispatch reset of the CU memory submodel (cold caches, full MSHR pool, idle BW
  // queues). cu_cycle and pipe state are monotonic and intentionally untouched.
  void                reset_memory() { mem_.reset(); rid_owner_.clear(); }

  // --- async shared-memory seam (R2.2c) ---------------------------------------
  // One issued memory instruction's L1-miss/bypass work. CuCycleModel drains
  // this queue each edge and sends it as MemReqMsg to MemSysCycleModel over a
  // simdojo Link; completions arrive asynchronously via on_mem_completion().
  struct MemReq { MemReqId rid = 0; std::vector<SharedReq> shared; };

  // Drain the requests produced this cycle (the servicer empties this each edge).
  std::vector<MemReq>& mem_requests() { return mem_out_; }

  // The servicer reports the shared-hierarchy completion cycle for `rid`. Resolves the
  // in-flight entry (complete_cyc), the deferred RAW mark, the held MSHRs, and the L1
  // fill readiness. Routes to the owning wave via rid_owner_.
  void on_mem_completion(MemReqId rid, uint64_t shared_complete_cyc);

  // --- memory-request lifecycle (model-owned; funcsim memory is sync) ----------
  // THE idempotent terminal. Found -> decrement the kind counter, tombstone, erase
  // (returns true). Not found -> classify via tombstones (dup/late/unknown counter)
  // and return false. The drain loop and the adapter's defensive paths share this.
  bool     complete_mem_req(CycleWaveState& ws, MemReqId rid);
  // Wave-kill teardown: leak-count every still-in-flight request, tombstone them,
  // and zero the outstanding counters. Funnel for every halt/trap path.
  void     drain_wave_at_halt(CycleWaveState& ws);
  // True for the memory pipe kinds {SMEM,VMEM,FLAT,GLOBAL,LDS} (adapter uses it to
  // decide whether to assign a wait-counter slot).
  static bool is_memory(InstrKind k);
  // Earliest model-computed completion across one wave / all resident waves.
  static uint64_t earliest_mem_complete(const CycleWaveState& ws);
  uint64_t earliest_mem_complete_all() const;

  const UarchConfig& cfg()   const { return cfg_; }
  CUState&           cu()           { return cu_; }
  SIMDUnitState&     simd(uint32_t i) { return simds_[i]; }
  uint32_t           num_simds() const { return cfg_.simds_per_cu; }

 private:
  // Shared issue core for the FIFO head AND the inline fast path. Mutates
  // scoreboard/pipe/front-end on ISSUED; never touches ws.pending. On a BLOCKED_*
  // result writes the strictly-future retry cycle to `retry`.
  IssueResult attempt_issue(CycleWaveState& ws, const InstrEvent& inst, uint64_t now, uint64_t& retry);

  // Config PipeSpec to fall back on when an opcode is absent from opcode_latency.
  // Routes per pipe so memory ops get their (large) pipe latency, not VALU's.
  const PipeSpec& fallback_spec(InstrKind k) const;
  static MemKind  mem_kind_of(InstrKind k, bool has_dst);
  // Complete every in_flight request whose complete_cyc has been reached, across
  // all resident waves. Called at the top of each tick_cycle / flush iteration.
  void            retire_due_mem();

  UarchConfig                cfg_;
  MemorySystem               mem_;   // per-CU memory submodel (init after cfg_)
  CUState                    cu_;
  std::vector<SIMDUnitState> simds_;
  std::vector<MemReq>        mem_out_;     // shared work produced this cycle (servicer drains)
  std::unordered_map<MemReqId, CycleWaveState*> rid_owner_;  // routes completions to owning wave
};

}  // namespace cycle_model
