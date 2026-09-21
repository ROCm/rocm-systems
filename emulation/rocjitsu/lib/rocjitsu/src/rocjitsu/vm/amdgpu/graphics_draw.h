// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_GRAPHICS_DRAW_H_
#define ROCJITSU_VM_AMDGPU_GRAPHICS_DRAW_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/graphics_stage.h"
#include "rocjitsu/vm/amdgpu/pm4.h"

namespace rocjitsu::amdgpu {

/// Register snapshot and shader outputs for one ordered graphics draw.
class GraphicsDraw final : public GraphicsStage {
public:
  GraphicsDraw(const Pm4QueueState &state, rj_code_arch_t arch, uint32_t vertices);
  DispatchEntry vertex_dispatch() const;
  void initialize(Wavefront &wave, uint32_t workgroup, uint32_t wave_index) override;
  void export_lane(Wavefront &wave, uint32_t lane, uint32_t target, uint32_t mask,
                   const std::array<uint32_t, 4> &values) override;
  void finish_vertices();

private:
  rj_code_arch_t arch_;
  uint32_t vertex_count_;
  uint32_t primitive_type_;
  std::array<uint32_t, 0x400> sh_;
  std::array<uint32_t, 0x2000> context_;
  std::array<std::array<uint32_t, 4>, 64> positions_{};
  std::array<uint32_t, 64> position_masks_{};
  std::array<uint32_t, 64> layer_viewport_{};
  std::array<uint32_t, 64> primitives_{};
  std::array<bool, 64> primitive_valid_{};
};

} // namespace rocjitsu::amdgpu

#endif
