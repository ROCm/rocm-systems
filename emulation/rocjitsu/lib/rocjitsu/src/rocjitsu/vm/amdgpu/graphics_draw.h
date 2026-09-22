// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_GRAPHICS_DRAW_H_
#define ROCJITSU_VM_AMDGPU_GRAPHICS_DRAW_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/graphics_stage.h"
#include "rocjitsu/vm/amdgpu/pm4.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace rocjitsu::amdgpu {
class GpuVmAccess;

/// Register snapshot and shader outputs for one ordered graphics draw.
class GraphicsDraw final : public GraphicsStage {
public:
  GraphicsDraw(const Pm4QueueState &state, rj_code_arch_t arch, uint32_t vertices,
               std::vector<uint32_t> indices = {});
  DispatchEntry vertex_dispatch() const;
  void initialize(Wavefront &wave, uint32_t workgroup, uint32_t wave_index) override;
  void export_mask(Wavefront &wave, uint64_t mask) override;
  void export_lane(Wavefront &wave, uint32_t lane, uint32_t target, uint32_t mask,
                   const std::array<uint32_t, 4> &values) override;
  /// Advance only after the preceding shader dispatch has retired and caches are flushed.
  std::optional<DispatchEntry> advance(const GpuVmAccess &memory);
  bool fragment_stage() const { return fragment_stage_; }

private:
  rj_code_arch_t arch_;
  uint32_t vertex_count_;
  uint32_t total_vertices_, first_vertex_ = 0;
  uint32_t instance_count_, instance_ = 0;
  uint32_t primitive_type_;
  std::vector<uint32_t> indices_;
  std::array<uint32_t, 0x400> sh_;
  std::array<uint32_t, 0x2000> context_;
  std::array<std::array<uint32_t, 4>, 64> positions_{};
  std::array<uint32_t, 64> position_masks_{};
  std::array<uint32_t, 64> layer_viewport_{};
  std::array<uint32_t, 64> primitives_{};
  std::array<bool, 64> primitive_valid_{};
  struct Fragment {
    int32_t x = 0, y = 0;
    float i = 0, j = 0, z = 0;
    float linear_i = 0, linear_j = 0;
    std::array<float, 3> pull_model{};
    bool covered = false;
    uint32_t mask = 0;
    std::array<uint32_t, 4> color{};
  };
  struct FragmentWave {
    std::array<Fragment, 64> lanes{};
    std::vector<uint32_t> parameters;
    uint32_t color_layer = 0;
  };
  std::vector<FragmentWave> fragments_;
  bool fragment_stage_ = false;
  uint32_t fragment_wave_size_ = 0;
  uint32_t color_format_ = 0;
  uint32_t memory_format_ = 0;
  uint32_t color_bytes_ = 0;
  uint32_t width_ = 0, height_ = 0, swizzle_ = 0;
  uint64_t color_base_ = 0, color_slice_size_ = 0;
  uint32_t color_first_layer_ = 0, color_last_layer_ = 0;
  uint32_t color_max_mip_ = 0, color_mip_ = 0;
  uint32_t color_pitch_ = 0, color_tail_x_ = 0, color_tail_y_ = 0;
  bool color_enabled_ = false;
  uint32_t depth_control_ = 0;
  uint32_t depth_width_ = 0, depth_height_ = 0, depth_swizzle_ = 0;
  uint32_t depth_bytes_ = 0;
  uint64_t depth_base_ = 0;
  std::optional<uint64_t> color_metadata_, depth_metadata_;
  uint32_t depth_clear_ = 0;
  bool attachments_prepared_ = false;
  void finish_vertices();
  DispatchEntry fragment_dispatch() const;
  void rasterize(const GpuVmAccess &memory);
  void write_outputs(const GpuVmAccess &memory);
  uint32_t primitive_count() const;
  void select_vertex_group();
  std::optional<DispatchEntry> next_vertex_group();
};

} // namespace rocjitsu::amdgpu

#endif
