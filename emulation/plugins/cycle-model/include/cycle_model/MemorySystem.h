// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file MemorySystem.h
/// @brief Per-CU memory submodel: coalescing + set-assoc L1/L2 tags + MSHRs +
/// L2/HBM bandwidth queues + LDS bank conflict. One instance per ArchModel (CU).
///
/// `access()` is the mutating latency walk on the admitted issue path; it advances
/// tag/MSHR/queue state so later accesses observe the contention. `admit_probe()`
/// is the read-only MSHR admission gate (§2/§4c) — it mutates nothing.

#pragma once

#include "cycle_model/CacheModels.h"
#include "cycle_model/InstrEvent.h"
#include "cycle_model/UarchConfig.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cycle_model {

struct CycleWaveState;

class MemorySystem {
 public:
  explicit MemorySystem(const UarchConfig& cfg) : cfg_(cfg) {
    l1v_.init(cfg.l1v);
    l1s_.init(cfg.l1s);
    mshr_free_.assign(cfg.mshrs_per_l1v, 0);   // all MSHRs free at cycle 0
    mshr_owner_.assign(cfg.mshrs_per_l1v, nullptr);
  }

  // cfg_ is a reference into the owning ArchModel's UarchConfig, so MemorySystem
  // must never be copied or moved — a move would copy the reference verbatim and
  // leave it dangling to the source object. This pins ArchModel too (it holds a
  // MemorySystem by value), forcing the adapter's cu_models_ map to construct each
  // model node-stably in place (try_emplace) rather than move a temporary in.
  MemorySystem(const MemorySystem&) = delete;
  MemorySystem& operator=(const MemorySystem&) = delete;
  MemorySystem(MemorySystem&&) = delete;
  MemorySystem& operator=(MemorySystem&&) = delete;

  // Sentinel for an L1 fill / MSHR whose completion cycle is not yet known (async).
  static constexpr uint64_t kUnresolved = UINT64_MAX;

  struct AccessResult {
    uint64_t              local_ready_cyc = 0;   // absolute cyc from L1-resolved (hit/pending/LDS/flat) lines
    std::vector<SharedReq> shared;               // lines needing the shared hierarchy
    std::vector<uint32_t>  mshr_slots;           // MSHR slots claimed (held; freed at completion)
    std::vector<uint64_t>  l1_fill_lines;        // cold-miss L1 bases installed pending (resolve at completion)
  };

  /// L1-local memory walk. Resolves hits/pending/LDS/flat into local_ready_cyc; emits
  /// miss/NT/UC lines as SharedReqs; claims (and HOLDS) MSHRs for cold vector misses;
  /// installs cold-miss L1 victims with an unresolved ready_cyc. Does NOT touch L2/HBM.
  AccessResult access(const InstrEvent& inst, uint64_t now, CycleWaveState& ws);

  /// Resolve the deferred state once the shared hierarchy reports `complete`: free the
  /// held MSHR slots (set their free cycle) and set each pending L1 fill line ready in
  /// the cache the access targeted (`is_smem` => l1s_, else l1v_; one access = one cache).
  void resolve_fills(const std::vector<uint32_t>& mshr_slots,
                     const std::vector<uint64_t>& l1_fill_lines, bool is_smem, uint64_t complete);

  /// Read-only admission probe. Returns the earliest cycle the access can admit
  /// (<= now means "admit now"). Mutates nothing.
  uint64_t admit_probe(const InstrEvent& inst, uint64_t now, const CycleWaveState& ws) const;

  /// Release every MSHR claimed by `owner` (a halting wave): free its fills at `now`
  /// and drop the ownership token, so the entries no longer block live waves until a
  /// fill that will never arrive. Called from ArchModel::drain_wave_at_halt.
  void release_wave_mshrs(const CycleWaveState* owner, uint64_t now);

  /// Cross-dispatch reset (reset-per-dispatch, v0): invalidate all cache lines, free
  /// the whole MSHR pool, and rewind the L2/HBM bandwidth queues so each kernel starts
  /// cold. Cache/queue geometry (from config) is preserved. Does NOT touch any cycle
  /// counter — cu_cycle is monotonic and lives on CUState.
  void reset() {
    l1v_.reset(); l1s_.reset();
    std::fill(mshr_free_.begin(), mshr_free_.end(), 0);
    std::fill(mshr_owner_.begin(), mshr_owner_.end(), nullptr);
  }

 private:
  // LDS bank-conflict latency: lds_pipe.base_latency + (max distinct words in any
  // bank - 1). Same-address lanes broadcast (no conflict).
  uint64_t lds_latency(const MemAccess& m) const;

  // Vector-L1 cold misses need a free MSHR. Count the would-miss (tag-absent) lines.
  uint32_t vector_misses_needing_mshr(const InstrEvent& inst) const;
  static bool is_vector_mem(InstrKind k) {
    return k == InstrKind::VMEM || k == InstrKind::FLAT || k == InstrKind::GLOBAL;
  }

  const UarchConfig&    cfg_;
  TimedTagCache         l1v_;        // vector L1 (VMEM/FLAT/GLOBAL)
  TimedTagCache         l1s_;        // scalar L1 (SMEM)
  std::vector<uint64_t> mshr_free_;  // L1V MSHR pool: per-entry free cycle
  std::vector<const CycleWaveState*> mshr_owner_;  // owning wave per MSHR (for halt release)
};

}  // namespace cycle_model
