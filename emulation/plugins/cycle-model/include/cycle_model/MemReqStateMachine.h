// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file MemReqStateMachine.h
/// @brief Model-owned memory request lifecycle and tombstone bookkeeping.
///
/// Funcsim memory is synchronous (rocjitsu MemoryPipeline::issue does init and
/// complete in one call). The cycle model owns the *cycle-domain* lifecycle: it
/// allocates a MemReqId when the scheduler commits a memop to a model pipe, the
/// cache/LDS/HBM submodels compute complete_cyc, and the request retires when
/// cu_cycle reaches complete_cyc. Tombstones keep duplicate/late terminals
/// classifiable after the in_flight entry is erased.
///
/// MemReqId is a per-CU monotonic counter (lives in CUState, never reset across
/// dispatches) — so tombstone collisions are structurally impossible and the
/// (dispatch_id, wf_id) scoping the plan once described is unnecessary.

#pragma once

#include "cycle_model/InstrEvent.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cycle_model {

using MemReqId = uint64_t;

enum class MemKind : uint8_t {
  VMEM_LOAD, VMEM_STORE, SMEM_LOAD, SMEM_STORE,
  LDS_READ, LDS_WRITE, FLAT_LOAD, FLAT_STORE,
};

struct MemReqEntry {
  MemKind     kind = MemKind::VMEM_LOAD;
  WaitCounter wcnt = WaitCounter::VMCNT;   // counter slot to decrement on retire
  uint64_t    issue_cyc = 0;
  uint64_t    complete_cyc = 0;            // model-computed terminal; UINT64_MAX = unresolved (async)
  uint64_t    local_ready_cyc = 0;         // completion from L1-resolved lines
  InstrRegSet write_regs;                  // dst regs to RAW-mark at completion (deferred)
  std::vector<uint32_t> mshr_slots;        // held MSHR slots to free at completion
  std::vector<uint64_t> l1_fill_lines;     // cold-miss L1 line bases to resolve ready_cyc at completion
  bool        is_smem = false;             // routes resolve_fills to l1s_ (else l1v_); one access = one cache
};

/// Bounded tombstone set — fixed-capacity, arbitrary eviction (correctness does
/// not depend on which old rid is dropped, only on a recent window staying live).
class TombstoneSet {
 public:
  explicit TombstoneSet(uint32_t capacity = 256) : capacity_(capacity) {}
  void insert(MemReqId rid) {
    if (set_.size() >= capacity_) set_.erase(set_.begin());
    set_.insert(rid);
  }
  bool contains(MemReqId rid) const { return set_.count(rid) != 0; }
  void clear() { set_.clear(); }
 private:
  std::unordered_set<MemReqId> set_;
  uint32_t capacity_;
};

/// Counters surfaced as observational metrics — none alter modeled cycle counts.
struct MemReqCounters {
  uint64_t dup_completions = 0;
  uint64_t late_completion = 0;
  uint64_t terminal_unknown = 0;
  uint64_t leak_at_halt = 0;
};

}  // namespace cycle_model
