// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file SharedMemModel.h
/// @brief Shared L2 + HBM timing: set-assoc L2 tags with fill-readiness, the L2
/// bandwidth queue, and per-channel HBM bandwidth queues. Owned by MemSysCycleModel
/// (one per XCD/L2); reached over simdojo Links from each CuCycleModel. This is where
/// cross-CU L2/HBM contention serializes.

#pragma once

#include "cycle_model/CacheModels.h"
#include "cycle_model/UarchConfig.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace cycle_model {

class SharedMemModel {
 public:
  explicit SharedMemModel(const UarchConfig& cfg) : cfg_(cfg) {
    l2_.init(cfg.l2);
    l2_bw_.bytes_per_cycle = cfg.l2_bytes_per_cycle ? cfg.l2_bytes_per_cycle : 1;
    if (cfg.hbm_channels) {
      // Per-channel sustained rate is pre-computed in the JSON config as
      // hbm_bytes_per_channel_per_cycle (aggregate_gbs * period_ps / 1000 / channels).
      uint32_t per_ch = cfg.hbm_bytes_per_channel_per_cycle ? cfg.hbm_bytes_per_channel_per_cycle : 1;
      hbm_ch_.assign(cfg.hbm_channels, BwQueue{});
      for (auto& q : hbm_ch_) q.bytes_per_cycle = per_ch;
    }
  }

  // Holds a reference into the owner's UarchConfig (same contract as MemorySystem):
  // never copy or move.
  SharedMemModel(const SharedMemModel&) = delete;
  SharedMemModel& operator=(const SharedMemModel&) = delete;
  SharedMemModel(SharedMemModel&&) = delete;
  SharedMemModel& operator=(SharedMemModel&&) = delete;

  bool l2_configured() const { return l2_.configured(); }

  // The physical L2 line base for an L1-line address (for the caller's per-L2-line
  // dedup). When no L2 is modeled (test cfg), the address is its own key.
  uint64_t l2_line_base(uint64_t base) const {
    return l2_.configured() ? (base & ~static_cast<uint64_t>(l2_.line_bytes - 1)) : base;
  }

  // Completion cycle for one (L1-line-sized) request that reaches the shared hierarchy
  // at `arrive_cyc`. `skip_l2` (UC) bypasses L2 straight to HBM; otherwise the normal
  // L2 walk (hit / pending-fill coalesce / miss -> HBM + install tag). Mutates L2 tags
  // and the L2/HBM bandwidth queues — this is the cross-CU contention serialization.
  uint64_t access_line(uint64_t l1_line_base, uint64_t arrive_cyc, bool skip_l2) {
    return skip_l2 ? uc_access(l1_line_base, arrive_cyc) : l2_access(l1_line_base, arrive_cyc);
  }

  // Service a memory instruction's shared-hierarchy lines: dedup by physical L2 line
  // (book each L2 line's bandwidth once), return the max completion cycle. 0 if empty.
  uint64_t service(const std::vector<SharedReq>& reqs) {
    uint64_t done = 0;
    std::vector<std::pair<uint64_t, uint64_t>> seen;   // l2base -> line_done
    for (const SharedReq& r : reqs) {
      uint64_t l2base = l2_line_base(r.line_base);
      uint64_t d = 0;
      bool hit = false;
      for (auto& [b, v] : seen) if (b == l2base) { d = v; hit = true; break; }
      if (!hit) { d = access_line(r.line_base, r.arrive_cyc, r.skip_l2); seen.emplace_back(l2base, d); }
      done = std::max(done, d);
    }
    return done;
  }

  // Cross-dispatch reset: invalidate L2, rewind the L2 + HBM bandwidth queues.
  void reset() {
    l2_.reset();
    l2_bw_.reset();
    for (auto& q : hbm_ch_) q.reset();
  }

 private:
  uint64_t l2_access(uint64_t l1_line_base, uint64_t arrive_cyc);
  uint64_t hbm_access(uint64_t l2base, uint64_t arrive);
  uint64_t uc_access(uint64_t line_base, uint64_t arrive);

  const UarchConfig&    cfg_;
  TimedTagCache         l2_;       // shared L2 tags + fill-readiness
  BwQueue               l2_bw_;    // L2 bandwidth queue
  std::vector<BwQueue>  hbm_ch_;   // per-HBM-channel bandwidth queues
};

}  // namespace cycle_model
