// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file WaveScoreboard.h
/// @brief Per-wavefront implicit-register RAW readiness tracker.
///
/// Tracks the CU cycle at which each register's most recent producer retires.
/// earliest_issue scans the instruction's READ set only — AMDGPU hardware has no
/// write-after-write interlock (cross-pipe WAW is the compiler's job via
/// s_waitcnt), so scanning writes here would invent stalls real silicon does not
/// have. mark_writes records future free_cyc for downstream RAW consumers.
///
/// Storage is per-wavefront (lives in CycleWaveState). The VGPR/SGPR vectors are
/// sized to the wave's register allocation; special registers get dedicated slots.

#pragma once

#include "cycle_model/InstrEvent.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cycle_model {

class Scoreboard {
 public:
  void resize(uint32_t num_vgprs, uint32_t num_sgprs) {
    vgpr_free_cyc_.assign(num_vgprs, 0);
    sgpr_free_cyc_.assign(num_sgprs, 0);
  }

  void clear() {
    std::fill(vgpr_free_cyc_.begin(), vgpr_free_cyc_.end(), 0);
    std::fill(sgpr_free_cyc_.begin(), sgpr_free_cyc_.end(), 0);
    vcc_free_cyc_ = exec_free_cyc_ = scc_free_cyc_ = m0_free_cyc_ = 0;
  }

  /// Earliest CU cycle the wave can issue an instruction whose source operands
  /// are listed in `rw`, given the wave is otherwise eligible at `now`.
  /// READ-ONLY scan — matches AMDGPU's lack of a HW WAW scoreboard.
  uint64_t earliest_issue(const InstrRegSet& rw, uint64_t now) const {
    uint64_t t = now;
    for (Reg r : rw.vgprs_read) t = std::max(t, vgpr_free_cyc_[r]);
    for (Reg r : rw.sgprs_read) t = std::max(t, sgpr_free_cyc_[r]);
    if (rw.reads_vcc)  t = std::max(t, vcc_free_cyc_);
    if (rw.reads_exec) t = std::max(t, exec_free_cyc_);
    if (rw.reads_scc)  t = std::max(t, scc_free_cyc_);
    if (rw.reads_m0)   t = std::max(t, m0_free_cyc_);
    return t;
    // Deliberately NO scan over vgprs_written / sgprs_written.
  }

  /// Record that this instruction's destination registers become readable at
  /// `retire_cyc`. Feeds future earliest_issue calls of downstream readers (RAW).
  void mark_writes(const InstrRegSet& rw, uint64_t retire_cyc) {
    for (Reg r : rw.vgprs_written) vgpr_free_cyc_[r] = retire_cyc;
    for (Reg r : rw.sgprs_written) sgpr_free_cyc_[r] = retire_cyc;
    if (rw.writes_vcc)  vcc_free_cyc_  = retire_cyc;
    if (rw.writes_exec) exec_free_cyc_ = retire_cyc;
    if (rw.writes_scc)  scc_free_cyc_  = retire_cyc;
    if (rw.writes_m0)   m0_free_cyc_   = retire_cyc;
  }

 private:
  std::vector<uint64_t> vgpr_free_cyc_;
  std::vector<uint64_t> sgpr_free_cyc_;
  uint64_t vcc_free_cyc_ = 0, exec_free_cyc_ = 0, scc_free_cyc_ = 0, m0_free_cyc_ = 0;
};

}  // namespace cycle_model
