// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/pm4/pm4.h"
#include "rocjitsu/vm/graphics/gs_registers.h"

namespace rocjitsu::amdgpu {

class GraphicsDraw;

/// State retained across draws of one graphics command stream.
struct GraphicsCommandState {
  uint64_t index_base = 0;
  uint32_t index_buffer_size = 0;
  struct IndirectDraw {
    uint64_t arguments = 0;
    uint32_t count = 0, next = 0, stride = 0;
    uint32_t vertex_register = 0, instance_register = 0;
    std::optional<uint32_t> first_index_register, draw_index_register;
  };
  std::optional<IndirectDraw> indirect_draw;
  uint32_t num_instances = 1;
  // Counter events are engine state, separate from the context register bank.
  bool performance_counters_active = false;
  bool unsupported_pixel_counter_mode = false;
  uint64_t pixel_counter_instances = 0, occlusion_samples = 0;
  std::shared_ptr<GraphicsDraw> draw;
  std::shared_ptr<GsRegisters> gs_registers = std::make_shared<GsRegisters>();
  /// One saved context bank; nested CLEAR_STATE pushes are unsupported.
  std::unique_ptr<ComputeCommandState::ContextRegisters> saved_context_registers;
};

inline GraphicsCommandState &graphics_state(ComputeCommandState &state) {
  if (!state.graphics)
    state.graphics = std::make_shared<GraphicsCommandState>();
  return *state.graphics;
}

} // namespace rocjitsu::amdgpu
