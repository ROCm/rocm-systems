// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_GRAPHICS_GRAPHICS_DRAW_H_
#define ROCJITSU_VM_GRAPHICS_GRAPHICS_DRAW_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/pm4/pm4.h"
#include "rocjitsu/vm/graphics/graphics_stage.h"
#include "rocjitsu/vm/graphics/image_volume.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::amdgpu {
class GpuVmAccess;
class CpuDispatchPool;

/// Register snapshot and shader outputs for one ordered graphics draw.
class GraphicsDraw final : public GraphicsStage {
public:
  GraphicsDraw(const ComputeCommandState &state, rj_code_arch_t arch, uint32_t vertices,
               std::vector<uint32_t> indices = {});
  /// Select bounded independent primitive groups before the first shader launch.
  /// Unknown ring identity and ordered shader modes retain one-group scheduling.
  bool enable_vertex_batching(const GpuVmAccess &memory, uint32_t limit = 32);
  DispatchEntry vertex_dispatch() const;
  void initialize(Wavefront &wave, uint32_t workgroup, uint32_t wave_index) override;
  bool allocate_exports(Wavefront &wave, uint32_t vertices, uint32_t primitives) override;
  void export_mask(Wavefront &wave, uint64_t mask) override;
  void export_lane(Wavefront &wave, uint32_t lane, uint32_t target, uint32_t mask,
                   const std::array<uint32_t, 4> &values) override;
  /// Advance only after the preceding shader dispatch has retired and caches are flushed.
  /// RAM read batching requires a caller with no observers or active debugging.
  std::optional<DispatchEntry> advance(const GpuVmAccess &memory, CpuDispatchPool *pool = nullptr,
                                       uint32_t threads = 1, bool allow_ram_read_batching = false,
                                       bool allow_early_depth = false);
  bool fragment_stage() const { return fragment_stage_; }
  uint64_t occlusion_samples() const { return occlusion_samples_; }
  std::shared_ptr<GsRegisters> gs_registers() const override {
    return fragment_stage_ ? nullptr : gs_registers_;
  }

private:
  friend class GraphicsDrawTestAccess;
  static constexpr uint32_t kColorTargets = 8;
  rj_code_arch_t arch_;
  uint32_t total_vertices_, first_vertex_ = 0;
  uint32_t instance_count_, instance_ = 0;
  uint32_t primitive_type_;
  bool count_occlusion_ = false;
  uint64_t occlusion_samples_ = 0;
  bool geometry_shader_ = false;
  uint32_t geometry_vertices_ = 0;
  std::vector<uint32_t> indices_;
  std::array<uint32_t, 0x400> sh_;
  std::array<uint32_t, 0x2000> context_;
  uint64_t attribute_ring_base_;
  uint32_t attribute_ring_bytes_;
  // A negative only disables an optional path for this draw. It grants no
  // access, and mapping changes need not turn it back on.
  bool attribute_ram_candidate_ = false;
  std::shared_ptr<GsRegisters> gs_registers_;
  struct VertexGroup {
    uint32_t count = 0, first_vertex = 0, instance = 0, attribute_offset = 0;
    uint32_t output_vertices = 0, output_primitives = 0;
    std::array<std::array<uint32_t, 4>, 64> positions{};
    std::array<uint32_t, 64> position_masks{};
    std::array<uint32_t, 64> layer_viewport{};
    std::array<uint32_t, 64> primitives{};
    std::array<bool, 64> primitive_valid{};
  };
  std::vector<VertexGroup> vertex_groups_;
  uint32_t vertex_group_limit_ = 1, attribute_slot_bytes_ = 0;
  uint32_t next_raster_group_ = 0;
  struct ColorExport {
    uint32_t mask = 0;
    std::array<uint32_t, 4> values{};
  };
  struct Fragment {
    int32_t x, y;
    float i, j, z;
    float linear_i, linear_j;
    std::array<float, 3> pull_model;
    bool covered;
    uint8_t stencil_reference;
    bool stencil_exported;
  };
  class FragmentWave {
  public:
    struct InterpolatedLanes {};
    FragmentWave() noexcept : lanes{} {}
    // Only the reserved parallel raster batch may defer its live prefix.
    // Interpolation writes every field before the wave can move or be read.
    FragmentWave(InterpolatedLanes, uint32_t live_lanes) noexcept {
      for (std::size_t lane = live_lanes; lane < lanes.size(); ++lane)
        lanes[lane] = {};
    }
    std::array<Fragment, 64> lanes;
    std::vector<uint32_t> parameters;
    std::size_t export_offset = static_cast<std::size_t>(-1);
    uint32_t relative_layer = 0;
    bool front = true;
  };
  std::vector<FragmentWave> fragments_;
  // Color exports are private output state, separate from raster geometry.
  // Slots are fixed before FS publication; each wave owns 64 records per slot.
  std::vector<ColorExport> fragment_exports_;
  std::array<uint8_t, kColorTargets> export_slots_{};
  bool fragment_exports_prepared_ = false;
  // Keep original output order, including the late depth visit of omitted waves.
  std::vector<uint32_t> fragment_dispatch_indices_;
  bool fragment_selection_active_ = false;
  bool early_depth_state_ = false;
  bool fragment_stage_ = false;
  uint32_t fragment_wave_size_ = 0;
  struct ColorAttachment {
    uint32_t export_index = 0, export_format = 0, memory_format = 0, bytes = 0, components = 0;
    uint32_t width = 0, height = 0, swizzle = 0;
    uint64_t base = 0, slice_size = 0;
    uint32_t first_layer = 0, last_layer = 0;
    uint32_t max_mip = 0, mip = 0;
    uint32_t resource_width = 0, resource_height = 0;
    uint32_t pitch = 0, tail_x = 0, tail_y = 0;
    uint32_t write_mask = 0, blend = 0;
    std::array<uint32_t, 4> component_indices{0, 1, 2, 3};
    std::array<uint32_t, 4> component_widths{};
    uint32_t selectors = 0xfac, channel_mask = 15;
    std::optional<ImageVolumeMipLayout> volume;
    bool srgb = false, pipe_aligned = false;
    std::optional<uint64_t> metadata;
  };
  std::array<ColorAttachment, kColorTargets> colors_{};
  uint32_t width_ = 0, height_ = 0;
  bool color_enabled_ = false;
  uint32_t depth_control_ = 0;
  uint32_t depth_width_ = 0, depth_height_ = 0, depth_swizzle_ = 0;
  uint32_t depth_bytes_ = 0;
  uint64_t depth_base_ = 0;
  uint64_t depth_slice_size_ = 0;
  uint32_t depth_first_layer_ = 0, depth_last_layer_ = 0;
  uint32_t depth_pitch_ = 0, depth_tail_x_ = 0, depth_tail_y_ = 0;
  std::optional<uint64_t> depth_metadata_;
  uint32_t depth_clear_ = 0;
  bool depth_metadata_has_stencil_ = false;
  uint8_t stencil_clear_ = 0;
  uint64_t stencil_base_ = 0, stencil_slice_size_ = 0;
  uint32_t stencil_swizzle_ = 0, stencil_pitch_ = 0;
  uint32_t stencil_tail_x_ = 0, stencil_tail_y_ = 0;
  bool attachments_prepared_ = false;
  void finish_vertices(const VertexGroup &group);
  DispatchEntry fragment_dispatch() const;
  void prepare_fragment_exports();
  const ColorExport &fragment_export(const FragmentWave &batch, uint32_t lane,
                                     uint32_t target) const;
  FragmentWave &fragment_for_dispatch(uint32_t workgroup);
  bool select_fragment_waves(const GpuVmAccess &memory);
  void prepare_colors();
  void prepare_attachments();
  bool try_gather_attributes(const GpuVmAccess &memory, const VertexGroup &group,
                             const std::array<uint32_t, 3> &indices, uint32_t provoking_index,
                             std::span<uint32_t> words);
  bool classify_attribute_ram_reads() const;
  void rasterize(const GpuVmAccess &memory, const VertexGroup &group, CpuDispatchPool *pool,
                 uint32_t threads, bool allow_ram_read_batching);
  void write_outputs(const GpuVmAccess &memory, CpuDispatchPool *pool, uint32_t threads);
  bool try_parallel_outputs(const GpuVmAccess &memory, CpuDispatchPool &pool, uint32_t threads);
  template <typename Memory>
  void write_output_fragment(const Memory &memory, const FragmentWave &batch, uint32_t lane);
  uint32_t primitive_count(const VertexGroup &group) const;
  uint32_t attribute_stride() const {
    // Per-primitive parameters occupy ring space even when the PS only reads vertex inputs.
    return ((sh_[0x31] & 31) + 1 + ((sh_[0x31] >> 5) & 31)) * 16;
  }
  void select_vertex_groups();
  std::optional<DispatchEntry> next_vertex_group();
};

} // namespace rocjitsu::amdgpu

#endif
