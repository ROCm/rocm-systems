// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_GRAPHICS_STAGE_H_
#define ROCJITSU_VM_AMDGPU_GRAPHICS_STAGE_H_

#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu {

class Wavefront;

/// Inputs and export destinations shared by the waves of a graphics stage.
class GraphicsStage {
public:
  virtual ~GraphicsStage() = default;
  virtual void initialize(Wavefront &wave, uint32_t workgroup, uint32_t wave_index) = 0;
  /// Pixel validity accompanies exports even when no color components are enabled.
  virtual void export_mask(Wavefront &wave, uint64_t mask) = 0;
  virtual void export_lane(Wavefront &wave, uint32_t lane, uint32_t target, uint32_t mask,
                           const std::array<uint32_t, 4> &values) = 0;
};

} // namespace rocjitsu::amdgpu

#endif
