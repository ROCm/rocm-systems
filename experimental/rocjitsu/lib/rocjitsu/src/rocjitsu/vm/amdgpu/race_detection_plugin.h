// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/execution_plugin.h"

#include <race-emulator/RaceDetector.h>
#include <race-emulator/WaveRaceState.h>

#include <cstdint>
#include <map>
#include <span>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// @brief Race detection plugin for the execution pipeline.
///
/// Creates per-workgroup RaceDetectors, registers memory events, checks
/// for VGPR/SGPR/LDS races, and collects diagnostic strings.
class RaceDetectionPlugin : public ExecutionPlugin {
public:
  void onWorkgroupDispatched(uint32_t wg_id, uint32_t lds_size,
                             uint32_t vgpr_count, uint32_t sgpr_count,
                             std::span<Wavefront *> wavefronts) override;

  void onMemoryInstruction(Instruction *inst, Wavefront &wf) override;

  void onVgprRead(Wavefront *wf, uint32_t logical_reg, uint32_t lane) override;

  void onSgprRead(Wavefront *wf, uint32_t logical_reg) override;

  void onWaitcnt(Wavefront *wf, int vmcnt, int lgkmcnt) override;

  /// @brief Return the diagnostic strings, keyed by workgroup id.
  const std::map<uint32_t, std::vector<std::string>> &diagnostics() const {
    return diagnostics_;
  }

private:
  std::map<uint32_t, std::unique_ptr<raceemulator::RaceDetector>> detectors_;
  std::map<uint32_t, std::vector<std::string>> diagnostics_;
  std::map<Wavefront *, uint32_t> wavefront_to_index_;
  std::map<Wavefront *, raceemulator::WaveRaceState *> wavefront_to_race_state_;
};

} // namespace amdgpu
} // namespace rocjitsu
