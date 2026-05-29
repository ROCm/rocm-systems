// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Scheduler and memory-lifecycle bodies: non-advancing try_issue_head_at,
/// round-robin scheduler_step over SIMDs, tick_cycle per-edge driver,
/// and the model-owned memory-request lifecycle. Kept out of the header so
/// UarchConfig/json stay off the hot include path.

#include "cycle_model/ArchModel.h"

#include <algorithm>

namespace cycle_model {

const PipeSpec& ArchModel::fallback_spec(InstrKind k) const {
  switch (k) {
    case InstrKind::VALU:   return cfg_.valu;
    case InstrKind::SALU:   return cfg_.salu;
    case InstrKind::SMEM:   return cfg_.smem;
    case InstrKind::VMEM:
    case InstrKind::FLAT:
    case InstrKind::GLOBAL: return cfg_.vmem;
    case InstrKind::LDS:    return cfg_.lds_pipe;
    case InstrKind::MFMA:   return cfg_.mfma;
    case InstrKind::WMMA:   return cfg_.wmma;
    default:                return cfg_.valu;   // BRANCH/OTHER: scalar-ish default
  }
}

bool ArchModel::is_memory(InstrKind k) {
  switch (k) {
    case InstrKind::SMEM: case InstrKind::VMEM: case InstrKind::FLAT:
    case InstrKind::GLOBAL: case InstrKind::LDS: return true;
    default: return false;
  }
}

MemKind ArchModel::mem_kind_of(InstrKind k, bool has_dst) {
  switch (k) {
    case InstrKind::SMEM:   return has_dst ? MemKind::SMEM_LOAD  : MemKind::SMEM_STORE;
    case InstrKind::LDS:    return has_dst ? MemKind::LDS_READ   : MemKind::LDS_WRITE;
    case InstrKind::FLAT:   return has_dst ? MemKind::FLAT_LOAD  : MemKind::FLAT_STORE;
    case InstrKind::VMEM:
    case InstrKind::GLOBAL:
    default:                return has_dst ? MemKind::VMEM_LOAD  : MemKind::VMEM_STORE;
  }
}

bool ArchModel::complete_mem_req(CycleWaveState& ws, MemReqId rid) {
  auto it = ws.in_flight.find(rid);
  if (it == ws.in_flight.end()) {
    if      (ws.completed_rids.contains(rid)) ws.mem_counters.dup_completions++;
    else if (ws.halted_rids.contains(rid))    ws.mem_counters.late_completion++;
    else                                       ws.mem_counters.terminal_unknown++;
    return false;
  }
  ws.outstanding.decrement(it->second.wcnt);
  ws.completed_rids.insert(rid);
  ws.in_flight.erase(it);
  return true;
}

void ArchModel::drain_wave_at_halt(CycleWaveState& ws) {
  ws.mem_counters.leak_at_halt += ws.in_flight.size();
  for (auto& kv : ws.in_flight) { ws.halted_rids.insert(kv.first); rid_owner_.erase(kv.first); }
  ws.in_flight.clear();
  ws.outstanding.clear();
  // Release the CU MSHRs this wave still holds — their fills will never arrive, so a
  // lingering claim would spuriously block live waves' admission (review #4).
  mem_.release_wave_mshrs(&ws, cu_.cu_cycle);
  // Drop the wave from its SIMD's resident list — its CycleWaveState is freed when
  // the Wavefront plugin-state slot is reused on a later dispatch, so a lingering
  // pointer here would dangle for scheduler_step / tick_cycle / flush_to_quiescence.
  auto& waves = simds_[ws.simd_id].waves;
  waves.erase(std::remove(waves.begin(), waves.end(), &ws), waves.end());
}

uint64_t ArchModel::earliest_mem_complete(const CycleWaveState& ws) {
  uint64_t e = UINT64_MAX;
  for (auto& kv : ws.in_flight) e = std::min(e, kv.second.complete_cyc);
  return e;
}

uint64_t ArchModel::earliest_mem_complete_all() const {
  uint64_t e = UINT64_MAX;
  for (auto& s : simds_)
    for (auto* w : s.waves) e = std::min(e, earliest_mem_complete(*w));
  return e;
}

bool ArchModel::has_work() const {
  if (!mem_out_.empty()) return true;
  for (auto& s : simds_)
    for (auto* w : s.waves)
      if (!w->pending.empty() || !w->in_flight.empty()) return true;
  return false;
}

void ArchModel::retire_due_mem() {
  for (auto& s : simds_)
    for (auto* w : s.waves) {
      // Collect first — complete_mem_req erases from the map we are scanning.
      for (;;) {
        MemReqId due = 0;
        bool found = false;
        for (auto& kv : w->in_flight)
          if (kv.second.complete_cyc <= cu_.cu_cycle) { due = kv.first; found = true; break; }
        if (!found) break;
        complete_mem_req(*w, due);
      }
    }
}

IssueResult ArchModel::attempt_issue(CycleWaveState& ws, const InstrEvent& inst,
                                     uint64_t now, uint64_t& retry) {
  uint64_t we = ws.scoreboard.earliest_issue(inst.regs, now);
  if (we > now) {
    ws.next_ready_cyc = std::max(ws.next_ready_cyc, we);
    retry = we;
    return IssueResult::BLOCKED_SCOREBOARD;
  }
  SIMDUnitState& simd = simds_[ws.simd_id];
  IssueQueue& pipe = pick_pipe(inst.kind, simd);
  if (pipe.next_free_cyc > now) { retry = pipe.next_free_cyc; return IssueResult::BLOCKED_PIPE; }
  if (simd.front_end_accounting_cyc == now && simd.front_end_issues_used >= cfg_.front_end_issue_per_simd) {
    retry = now + 1;
    return IssueResult::BLOCKED_FRONT_END;
  }
  // Memory admission gate (§2/§4c): read-only MSHR probe. A would-miss with no free
  // MSHR cannot admit — block issue, retry at the earliest free cycle, mutate nothing.
  if (is_memory(inst.kind)) {
    uint64_t mshr_free = mem_.admit_probe(inst, now, ws);
    if (mshr_free > now) { retry = mshr_free; return IssueResult::BLOCKED_MSHR; }
  }
  // Memory ops: the L1-local walk resolves the synchronous part and emits any shared
  // work as deferred requests (no synchronous complete_cyc). Everything else uses the
  // per-opcode/per-pipe latency table and completes synchronously.
  if (is_memory(inst.kind)) {
    MemorySystem::AccessResult ar = mem_.access(inst, now, ws);
    bool has_dst = !inst.regs.vgprs_written.empty() || !inst.regs.sgprs_written.empty();
    MemReqId rid = cu_.alloc_mem_req_id();
    MemReqEntry e;
    e.kind = mem_kind_of(inst.kind, has_dst);
    e.wcnt = inst.wcnt;
    e.issue_cyc = now;
    e.local_ready_cyc = ar.local_ready_cyc;
    e.mshr_slots = std::move(ar.mshr_slots);
    e.l1_fill_lines = std::move(ar.l1_fill_lines);
    e.is_smem = (inst.kind == InstrKind::SMEM);  // routes resolve_fills to l1s_/l1v_
    ws.outstanding.increment(inst.wcnt);
    pipe.next_free_cyc = now + pipe.issue_rate;
    if (simd.front_end_accounting_cyc != now) { simd.front_end_accounting_cyc = now; simd.front_end_issues_used = 0; }
    simd.front_end_issues_used++;
    ws.next_ready_cyc = now + 1;
    if (ar.shared.empty()) {
      // L1 hit / LDS / flat: completes synchronously. RAW mark now.
      e.complete_cyc = ar.local_ready_cyc;
      ws.scoreboard.mark_writes(inst.regs, e.complete_cyc);
      ws.in_flight.emplace(rid, std::move(e));
    } else {
      // L1 miss / bypass: completion deferred to on_mem_completion. RAW mark deferred
      // too (load-use is waitcnt-gated).
      e.complete_cyc = UINT64_MAX;          // retire_due_mem skips until resolved
      e.write_regs = inst.regs;             // RAW-marked at async completion
      rid_owner_[rid] = &ws;
      ws.in_flight.emplace(rid, std::move(e));
      mem_out_.push_back({rid, std::move(ar.shared)});
    }
    return IssueResult::ISSUED;
  }

  // Non-memory: latency from the per-opcode/per-pipe table; completes synchronously.
  uint64_t lat_cyc = cfg_.lookup_latency(inst.mnemonic, fallback_spec(inst.kind)).base_latency;
  uint64_t complete_cyc = now + lat_cyc;
  ws.scoreboard.mark_writes(inst.regs, complete_cyc);
  pipe.next_free_cyc = now + pipe.issue_rate;
  if (simd.front_end_accounting_cyc != now) { simd.front_end_accounting_cyc = now; simd.front_end_issues_used = 0; }
  simd.front_end_issues_used++;
  ws.next_ready_cyc = now + 1;
  return IssueResult::ISSUED;
}

void ArchModel::on_mem_completion(MemReqId rid, uint64_t shared_complete_cyc) {
  auto own = rid_owner_.find(rid);
  if (own == rid_owner_.end()) return;          // already retired/halted: drop (idempotent)
  CycleWaveState& ws = *own->second;
  auto it = ws.in_flight.find(rid);
  if (it == ws.in_flight.end()) { rid_owner_.erase(own); return; }
  MemReqEntry& e = it->second;
  uint64_t complete = std::max(e.local_ready_cyc, shared_complete_cyc);
  e.complete_cyc = complete;                    // retire_due_mem fires when cu_cycle >= complete
  ws.scoreboard.mark_writes(e.write_regs, complete);          // deferred RAW mark
  mem_.resolve_fills(e.mshr_slots, e.l1_fill_lines, e.is_smem, complete);
  rid_owner_.erase(own);
}

IssueResult ArchModel::try_issue_inline(CycleWaveState& ws, const InstrEvent& inst, uint64_t now) {
  uint64_t retry = UINT64_MAX;
  return attempt_issue(ws, inst, now, retry);
}

IssueDecision ArchModel::try_issue_head_at(CycleWaveState& ws, uint64_t now) {
  if (ws.pending.empty()) return {IssueResult::EMPTY, UINT64_MAX};
  PendingEvent& e = ws.pending.front();
  if (!e.metadata_ready) return {IssueResult::BLOCKED_METADATA, UINT64_MAX};

  // Positional waitcnt gate: clears the moment the outstanding counters meet the
  // thresholds. Clearing is free (no pipe/front-end/cycle cost). While blocked, the
  // only thing that can lower a counter is a memory completion, so that is the retry.
  if (e.kind == PendingKind::WaitcntGate) {
    if (e.waitcnt.target.satisfied(ws.outstanding)) {
      ws.pending.pop_front();
      return {IssueResult::GATE_CLEARED, now};
    }
    return {IssueResult::BLOCKED_GATE, earliest_mem_complete(ws)};
  }

  // Positional barrier gate: clears only when the adapter has signalled this wave's
  // workgroup barrier resolved. No internal event unblocks it (retry == MAX).
  if (e.kind == PendingKind::BarrierGate) {
    if (ws.barrier_signals > 0) {
      ws.barrier_signals--;          // consume one resolve; multiple are queued in the passive path
      ws.pending.pop_front();
      return {IssueResult::GATE_CLEARED, now};
    }
    return {IssueResult::BLOCKED_GATE, UINT64_MAX};
  }

  uint64_t retry = UINT64_MAX;
  IssueResult r = attempt_issue(ws, e.instr, now, retry);
  if (r == IssueResult::ISSUED) { ws.pending.pop_front(); return {r, now + 1}; }
  return {r, retry};
}

SchedulerStepResult ArchModel::scheduler_step() {
  SchedulerStepResult result;
  bool any_progress;
  do {
    any_progress = false;
    for (uint32_t attempt = 0; attempt < cfg_.simds_per_cu; ++attempt) {
      uint32_t si = (cu_.next_simd_idx + attempt) % cfg_.simds_per_cu;
      SIMDUnitState& simd = simds_[si];
      if (simd.front_end_accounting_cyc != cu_.cu_cycle) {
        simd.front_end_accounting_cyc = cu_.cu_cycle;
        simd.front_end_issues_used = 0;
      }
      if (simd.front_end_issues_used >= cfg_.front_end_issue_per_simd) continue;
      if (simd.waves.empty()) continue;
      uint32_t n = static_cast<uint32_t>(simd.waves.size());
      for (uint32_t k = 0; k < n; ++k) {
        uint32_t i = (simd.next_wave_idx + k) % n;
        IssueDecision d = try_issue_head_at(*simd.waves[i], cu_.cu_cycle);
        if (d.result == IssueResult::ISSUED) {
          simd.next_wave_idx = (i + 1) % n;
          result.issues++;
          result.earliest_retry = std::min(result.earliest_retry, d.earliest_retry);
          any_progress = true;
          break;
        }
        if (d.result == IssueResult::GATE_CLEARED) {
          any_progress = true;   // free clear; re-examine this wave's new head
          break;
        }
        if (d.result == IssueResult::BLOCKED_SCOREBOARD || d.result == IssueResult::BLOCKED_PIPE ||
            d.result == IssueResult::BLOCKED_FRONT_END  || d.result == IssueResult::BLOCKED_GATE ||
            d.result == IssueResult::BLOCKED_MSHR)
          result.earliest_retry = std::min(result.earliest_retry, d.earliest_retry);
      }
    }
  } while (any_progress);
  cu_.next_simd_idx = (cu_.next_simd_idx + 1) % cfg_.simds_per_cu;
  return result;
}

bool ArchModel::tick_cycle() {
  retire_due_mem();                         // terminate completed memops first
  SchedulerStepResult step = scheduler_step();
  if (step.issues == 0 && has_work()) cu_.stalls.sched += 1;  // idle-but-blocked cycle
  cu_.cu_cycle += 1;
  return has_work();
}

void ArchModel::flush_to_quiescence() {
  // has_work() covers pending instructions, in_flight async memops, AND mem_out_ — so a
  // wave whose pending FIFO is empty but still has an outstanding shared-mem completion
  // keeps the loop alive long enough for retire_due_mem to drain its tail (review #7).
  while (has_work()) {
    retire_due_mem();
    SchedulerStepResult step = scheduler_step();   // internally drains same-cycle gate clears
    uint64_t next = UINT64_MAX;
    if (step.earliest_retry > cu_.cu_cycle) next = std::min(next, step.earliest_retry);
    uint64_t mc = earliest_mem_complete_all();
    if (mc > cu_.cu_cycle) next = std::min(next, mc);
    if (next == UINT64_MAX) {
      // No internal event can advance us. If scheduler_step also made no progress,
      // the only pending heads are externally-blocked barrier gates awaiting a
      // resolve signal — bail rather than spin +1 forever. (A later barrier-resolve
      // hook drives those waves; quiescence here means "nothing internal left".)
      if (step.issues == 0) break;
      next = cu_.cu_cycle + 1;
    }
    if (step.issues == 0) cu_.stalls.sched += (next - cu_.cu_cycle);
    cu_.cu_cycle = next;
  }
}

void ArchModel::drive_to_quiescence_passive() {
  // Alternate: drive what can issue now, then synchronously land the deferred async
  // memory that the (un-ticked) MemSys would otherwise have delivered. Each landed
  // request lowers an outstanding wait-counter, so the next flush clears the waitcnt
  // gate that was blocking the wave and issues its tail. Repeat until nothing is left
  // to issue AND nothing is left to land.
  for (;;) {
    flush_to_quiescence();
    if (mem_out_.empty()) break;   // no deferred async work left to land -> done
    // mem_out_ holds every shared request produced-but-never-sent (their in_flight
    // entries carry complete_cyc==UINT64_MAX). Land them at the current cycle.
    std::vector<MemReq> pending;
    pending.swap(mem_out_);
    for (auto& req : pending)
      on_mem_completion(req.rid, cu_.cu_cycle);  // sets complete_cyc, RAW mark, fills
  }
}

}  // namespace cycle_model
