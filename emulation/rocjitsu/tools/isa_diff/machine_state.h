// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file machine_state.h
/// @brief Complete initial/final state of one isolated region execution.
#pragma once

#include "json.h"
#include "location.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

/// Control state that P0 instructions neither read nor write. It is fixed to the
/// post-dispatch values, recorded, and checked for preservation after every case.
struct Frame {
  uint32_t status_except_scc = 0;
  uint32_t mode = 0;
  uint32_t trapsts = 0;
  std::array<uint32_t, 16> ttmp{};
  uint64_t scratch_base = 0;
  uint32_t s102 = 0; ///< Allocated SGPRs beyond s101 (not addressable as sN in P0).
  uint32_t s103 = 0;
  /// vmcnt, lgkmcnt, expcnt, vscnt, dscnt, kmcnt, tensorcnt, asynccnt, samplecnt.
  std::array<uint8_t, 9> wait_counters{};
  bool in_trap_handler = false;
  uint8_t wave_state = 0;
  uint32_t sleep_cycles = 0;

  bool operator==(const Frame &) const = default;
};

struct MachineState {
  std::array<uint32_t, kNumLocs> loc{}; ///< SCC is 0/1.
  Frame frame;
  /// VGPRs v0..v{vgpr_count-1}, register-major, kWaveLanes lanes each (VALU support).
  unsigned vgpr_count = 0;
  std::vector<uint32_t> vgpr;

  [[nodiscard]] uint32_t get(unsigned l) const { return loc[l]; }
  void set(unsigned l, uint32_t v) { loc[l] = v & loc_value_mask(l); }
  [[nodiscard]] uint32_t vget(unsigned r, unsigned lane) const {
    return vgpr[size_t{r} * kWaveLanes + lane];
  }
  void vset(unsigned r, unsigned lane, uint32_t v) { vgpr[size_t{r} * kWaveLanes + lane] = v; }
  void resize_vgprs(unsigned n) {
    vgpr_count = n;
    vgpr.assign(size_t{n} * kWaveLanes, 0);
  }

  [[nodiscard]] Json to_json() const;
  static std::optional<MachineState> from_json(const Json &j, std::string &error);
  bool operator==(const MachineState &) const = default;
};

/// Locations whose values differ.
[[nodiscard]] LocSet differing_locs(const MachineState &a, const MachineState &b);
/// VGPRs that differ on at least one lane (a count mismatch differs everywhere).
[[nodiscard]] VSet differing_vgprs(const MachineState &a, const MachineState &b);
/// Lanes of VGPR @p r that differ, as a 64-bit lane mask.
[[nodiscard]] uint64_t differing_lanes(const MachineState &a, const MachineState &b, unsigned r);
/// Frame field names (as in contract.h) whose values differ.
[[nodiscard]] std::vector<std::string> differing_frame_fields(const Frame &a, const Frame &b);

} // namespace rocjitsu::isa_diff
