// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/graphics_draw.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/image_address.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/log.h"

#include <bit>
#include <cmath>
#include <stdexcept>

namespace rocjitsu::amdgpu {
namespace {
constexpr uint32_t kTriangleList = 4, kTriangleStrip = 6, kRectangleList = 17;
}

GraphicsDraw::GraphicsDraw(const Pm4QueueState &state, rj_code_arch_t arch, uint32_t vertices)
    : arch_(arch), vertex_count_(vertices), total_vertices_(vertices),
      instance_count_(state.num_instances), primitive_type_(state.uconfig_registers[0x242]),
      sh_(state.sh_registers), context_(state.context_registers) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch != ROCJITSU_CODE_ARCH_RDNA4)
    throw std::runtime_error("graphics draw requires RDNA3 or RDNA4");
  if (!instance_count_ || instance_count_ > 4096 || vertices < 3 || vertices > (1u << 20) ||
      (primitive_type_ != kTriangleStrip && vertices % 3) ||
      (primitive_type_ != kTriangleList && primitive_type_ != kTriangleStrip &&
       primitive_type_ != kRectangleList))
    throw std::runtime_error("unsupported graphics primitive, vertex count, or instance count");
  select_vertex_group();
}

uint32_t GraphicsDraw::primitive_count() const {
  return primitive_type_ == kTriangleStrip ? vertex_count_ - 2 : vertex_count_ / 3;
}

void GraphicsDraw::select_vertex_group() {
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t wave = (context_[gfx12 ? 0x2a6 : 0x2d5] & (1u << 22)) ? 32 : 64;
  const uint32_t limit = primitive_type_ == kTriangleStrip ? wave : (wave / 3) * 3;
  vertex_count_ = std::min(total_vertices_ - first_vertex_, limit);
}

std::optional<DispatchEntry> GraphicsDraw::next_vertex_group() {
  if (first_vertex_ + vertex_count_ == total_vertices_) {
    first_vertex_ = 0;
    if (++instance_ == instance_count_)
      return std::nullopt;
  } else {
    first_vertex_ += primitive_type_ == kTriangleStrip ? vertex_count_ - 2 : vertex_count_;
  }
  select_vertex_group();
  positions_ = {};
  position_masks_ = {};
  layer_viewport_ = {};
  primitives_ = {};
  primitive_valid_ = {};
  fragments_.clear();
  fragment_stage_ = false;
  return vertex_dispatch();
}

DispatchEntry GraphicsDraw::vertex_dispatch() const {
  DispatchEntry dp;
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t rsrc1 = sh_[0x8a], rsrc2 = sh_[0x8b];
  dp.kernel_wave_size = (context_[gfx12 ? 0x2a6 : 0x2d5] & (1u << 22)) ? 32 : 64;
  if (vertex_count_ > dp.kernel_wave_size || (rsrc2 & 1))
    throw std::runtime_error("unsupported graphics wave count or scratch allocation");
  uint64_t pc = (uint64_t{sh_[gfx12 ? 0x86 : 0xc9]} << 32) | sh_[gfx12 ? 0x89 : 0xc8];
  pc <<= 8;
  dp.kernel_entry_pc = static_cast<uint64_t>(static_cast<int64_t>(pc << 16) >> 16);
  dp.vgprs_per_wf = ((rsrc1 & 63) + 1) * (dp.kernel_wave_size == 32 ? 8 : 4);
  dp.initial_mode_raw = (rsrc1 >> 12) & 0xff;
  if (rsrc1 & (1u << 31))
    dp.initial_mode_raw |= Wavefront::FP16_OVFL_BIT;
  dp.group_segment_fixed_size = ((rsrc2 >> 19) & 255) * 512;
  dp.wgp_mode = (rsrc1 >> 27) & 1;
  dp.total_wgs = dp.grid_wgs_x = dp.grid_wgs_y = dp.grid_wgs_z = 1;
  dp.grid_yz_valid = true;
  dp.grid_size_x = dp.workgroup_size_x = dp.kernel_wave_size;
  dp.grid_size_y = dp.grid_size_z = 1;
  dp.wait_for_predecessors = true;
  return dp;
}

void GraphicsDraw::initialize(Wavefront &wave, uint32_t workgroup, uint32_t wave_index) {
  if (fragment_stage_) {
    auto &batch = fragments_.at(wave.wg_coord()[0]);
    const uint32_t users = ((sh_[0xb] >> 1) & 31) | ((sh_[0xb] >> 22) & 32);
    for (uint32_t i = 0; i < users; ++i)
      wave.debug_write_sgpr(i, sh_[0xc + i]);
    // One primitive per wave: all quads share one set of LDS coefficients.
    wave.debug_write_sgpr(users, wave.lds_base());
    for (uint32_t i = 0; i < batch.parameters.size(); ++i)
      wave.lds().write32(wave.lds_base() + i * 4, batch.parameters[i]);
    uint64_t exec = 0;
    for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
      const auto &f = batch.lanes[lane];
      wave.debug_write_vgpr(0, lane, std::bit_cast<uint32_t>(f.i));
      wave.debug_write_vgpr(1, lane, std::bit_cast<uint32_t>(f.j));
      if (f.covered)
        exec |= uint64_t{1} << lane;
    }
    wave.set_exec(exec);
    return;
  }
  if (workgroup || wave_index)
    throw std::runtime_error("graphics wave outside supported primitive group");
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  // GFX11+ merged GS repurposes PGM_LO/HI_GS as the first two user SGPRs.
  wave.debug_write_sgpr(0, sh_[gfx12 ? 0x84 : 0x88]);
  wave.debug_write_sgpr(1, sh_[gfx12 ? 0x85 : 0x89]);
  wave.debug_write_sgpr(2, 0);
  wave.debug_write_sgpr(3, vertex_count_ | (primitive_count() << 8));
  for (uint32_t i = 4; i < 8; ++i)
    wave.debug_write_sgpr(i, 0);
  const uint32_t user_count = ((sh_[0x8b] >> 1) & 31) | ((sh_[0x8b] >> 22) & 32);
  // The merged-stage USER_SGPR count starts at s8; the ring pair is separate.
  for (uint32_t i = 0; i < user_count; ++i)
    wave.debug_write_sgpr(i + 8, sh_[0x8c + i]);
  const uint32_t index_bits = gfx12 ? 9 : 10;
  for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
    // Primitive connectivity is local to the primitive group's position exports.
    const uint32_t first = primitive_type_ == kTriangleStrip ? lane : 3 * lane;
    const bool reverse = primitive_type_ == kTriangleStrip && ((first_vertex_ + lane) & 1);
    wave.debug_write_vgpr(0, lane,
                          (first + reverse) | ((first + !reverse) << index_bits) |
                              ((first + 2) << (2 * index_bits)));
    for (uint32_t i = 1; i < (gfx12 ? 3u : 5u); ++i)
      wave.debug_write_vgpr(i, lane, 0);
    wave.debug_write_vgpr(gfx12 ? 3 : 5, lane, first_vertex_ + lane);
    const uint32_t components = (sh_[0x8b] >> 16) & 3;
    if (components >= (gfx12 ? 1u : 3u))
      wave.debug_write_vgpr(gfx12 ? 4 : 8, lane, instance_);
  }
}

void GraphicsDraw::export_lane(Wavefront &wave, uint32_t lane, uint32_t target, uint32_t mask,
                               const std::array<uint32_t, 4> &values) {
  if (fragment_stage_) {
    if (target != 0) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return;
    }
    auto &f = fragments_.at(wave.wg_coord()[0]).lanes[lane];
    for (uint32_t i = 0; i < 4; ++i)
      if (mask & (1u << i))
        f.color[i] = values[i];
    f.mask |= mask;
    return;
  }
  if (target == 12 && lane < vertex_count_) {
    for (uint32_t i = 0; i < 4; ++i)
      if (mask & (1u << i))
        positions_[lane][i] = values[i];
    position_masks_[lane] |= mask;
  } else if (target == 13 && lane < vertex_count_ && mask == 4) {
    layer_viewport_[lane] = values[2];
  } else if (target == 20 && lane < primitive_count() && mask == 1) {
    primitives_[lane] = values[0];
    primitive_valid_[lane] = true;
  } else {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
  }
}

void GraphicsDraw::finish_vertices() {
  for (uint32_t i = 0; i < primitive_count(); ++i)
    if (!primitive_valid_[i])
      throw std::runtime_error("missing graphics primitive export");
  for (uint32_t i = 0; i < vertex_count_; ++i) {
    if (layer_viewport_[i])
      throw std::runtime_error("graphics layer or viewport selection is not implemented");
    if (position_masks_[i] != 15)
      throw std::runtime_error("missing graphics position export");
    util::Logger::vm("position ", i, ": ", std::bit_cast<float>(positions_[i][0]), ", ",
                     std::bit_cast<float>(positions_[i][1]), ", ",
                     std::bit_cast<float>(positions_[i][2]), ", ",
                     std::bit_cast<float>(positions_[i][3]));
  }
}

DispatchEntry GraphicsDraw::fragment_dispatch() const {
  DispatchEntry dp;
  const uint32_t rsrc1 = sh_[0xa], rsrc2 = sh_[0xb];
  if (rsrc2 & 1)
    throw std::runtime_error("fragment scratch is not implemented");
  dp.kernel_wave_size = fragment_wave_size_;
  dp.kernel_entry_pc =
      addr_calc::buffer_virtual_address(((uint64_t{sh_[0x9]} << 32) | sh_[0x8]) << 8);
  dp.vgprs_per_wf = ((rsrc1 & 63) + 1) * (dp.kernel_wave_size == 32 ? 8 : 4);
  dp.initial_mode_raw = (rsrc1 >> 12) & 255;
  if (rsrc1 & (1u << 29))
    dp.initial_mode_raw |= Wavefront::FP16_OVFL_BIT;
  dp.group_segment_fixed_size = 2048;
  dp.total_wgs = dp.grid_wgs_x = fragments_.size();
  dp.grid_wgs_y = dp.grid_wgs_z = 1;
  dp.grid_yz_valid = true;
  dp.workgroup_size_x = dp.kernel_wave_size;
  dp.grid_size_x = dp.total_wgs * dp.kernel_wave_size;
  dp.grid_size_y = dp.grid_size_z = 1;
  dp.wait_for_predecessors = true;
  return dp;
}

void GraphicsDraw::rasterize(GpuMemory &memory, uint32_t process_id) {
  if (arch_ != ROCJITSU_CODE_ARCH_RDNA4)
    throw std::runtime_error("graphics rasterization currently requires RDNA4");
  const uint32_t info = context_[0x3b0];
  const uint32_t attrib = context_[0x31b], attrib2 = context_[0x31e], attrib3 = context_[0x31f];
  color_format_ = context_[0x195];
  util::Logger::cp("graphics target info=", std::hex, info, " attrib=", attrib, ",", attrib2, ",",
                   attrib3, " export=", color_format_, " inputs=", context_[0x198],
                   " mask=", context_[0x214], " depth=", context_[0x1c], std::dec);
  memory_format_ = ((info >> 8) & 7) == 4 ? 46 : 42;
  if ((info & 31) != 10 || (((info >> 8) & 31) != 0 && ((info >> 8) & 31) != 4) || attrib ||
      (attrib3 & 0xf83fff) || context_[0x31a] || context_[0x319] ||
      (context_[0x1e0] & (1u << 30)) || context_[0x1c] || context_[0x214] != 15 ||
      (color_format_ != 4 && color_format_ != 7 && color_format_ != 9) || context_[0x198] != 2)
    throw std::runtime_error("unsupported graphics color, depth, blend, or fragment inputs");
  width_ = (attrib2 >> 16) + 1;
  height_ = (attrib2 & 0xffff) + 1;
  if (width_ > 4096 || height_ > 4096)
    throw std::runtime_error("graphics render target exceeds supported dimensions");
  swizzle_ = (attrib3 >> 15) & 7;
  if (swizzle_ > 4)
    throw std::runtime_error("unsupported graphics surface layout");
  color_base_ = addr_calc::buffer_virtual_address(
      ((uint64_t{context_[0x390] & 255} << 32) | context_[0x318]) << 8);
  fragment_wave_size_ = (context_[0x190] & (1u << 15)) ? 32 : 64;
  const uint32_t attributes = (sh_[0x31] >> 11) & 63;
  if (attributes > 32 || (sh_[0x31] >> 17))
    throw std::runtime_error("unsupported graphics parameter count");
  std::array<uint32_t, 4> ring{};
  if (attributes) {
    const uint64_t table = (uint64_t{sh_[0x85]} << 32) | sh_[0x84];
    if (memory.read_block_exact(table + 0xa0,
                                {reinterpret_cast<uint8_t *>(ring.data()), sizeof(ring)},
                                process_id) != AccessOutcome::Complete)
      throw std::runtime_error("graphics attribute descriptor read failed");
  }
  const uint64_t ring_base =
      addr_calc::buffer_virtual_address((uint64_t{ring[1] & 0xffff} << 32) | ring[0]);
  const uint32_t ring_stride = ((sh_[0x31] & 31) + 1) * 16;
  const float sx = std::bit_cast<float>(context_[0x10f]);
  const float ox = std::bit_cast<float>(context_[0x110]);
  const float sy = std::bit_cast<float>(context_[0x111]);
  const float oy = std::bit_cast<float>(context_[0x112]);
  if (context_[0x205] != 0x43f || !std::isfinite(sx) || !std::isfinite(sy) || !std::isfinite(ox) ||
      !std::isfinite(oy))
    throw std::runtime_error("unsupported graphics viewport transform");
  const int left = std::max(0, int(context_[0x90] & 0x7fff));
  const int top = std::max(0, int((context_[0x90] >> 16) & 0x7fff));
  const int right = std::min(int(width_), int(context_[0x91] & 0x7fff));
  const int bottom = std::min(int(height_), int((context_[0x91] >> 16) & 0x7fff));
  for (uint32_t p = 0; p < primitive_count(); ++p) {
    if (primitives_[p] & (1u << 31))
      continue;
    std::array<uint32_t, 3> indices{};
    struct Point {
      double x, y, w;
    };
    std::array<Point, 3> v{};
    for (uint32_t k = 0; k < 3; ++k) {
      indices[k] = (primitives_[p] >> (9 * k)) & 511;
      if (indices[k] >= vertex_count_)
        throw std::runtime_error("graphics primitive index exceeds vertex exports");
      const auto &position = positions_[indices[k]];
      const float w = std::bit_cast<float>(position[3]);
      const float x = std::bit_cast<float>(position[0]);
      const float y = std::bit_cast<float>(position[1]);
      if (!std::isfinite(w) || w <= 0 || !std::isfinite(x) || !std::isfinite(y)) {
        throw std::runtime_error("graphics homogeneous clipping is not implemented");
      }
      v[k] = {std::round((x / w * sx + ox) * 256) / 256, std::round((y / w * sy + oy) * 256) / 256,
              w};
      if (!std::isfinite(v[k].x) || !std::isfinite(v[k].y) || std::abs(v[k].x) > (1 << 20) ||
          std::abs(v[k].y) > (1 << 20))
        throw std::runtime_error("graphics vertex exceeds supported raster coordinates");
    }
    const auto edge = [](Point a, Point b, double x, double y) {
      return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
    };
    const double area = edge(v[0], v[1], v[2].x, v[2].y);
    if (area == 0)
      continue;
    const bool front = (area < 0) != bool(context_[0x207] & 4);
    if ((front && (context_[0x207] & 1)) || (!front && (context_[0x207] & 2)))
      continue;
    const bool rectangle = primitive_type_ == kRectangleList;
    const int min_x = std::max(left, int(std::floor(std::min({v[0].x, v[1].x, v[2].x}))));
    const int min_y = std::max(top, int(std::floor(std::min({v[0].y, v[1].y, v[2].y}))));
    const int max_x = std::min(right, int(std::ceil(std::max({v[0].x, v[1].x, v[2].x}))));
    const int max_y = std::min(bottom, int(std::ceil(std::max({v[0].y, v[1].y, v[2].y}))));
    std::vector<uint32_t> parameters(attributes * 12);
    for (uint32_t a = 0; a < attributes; ++a) {
      const uint32_t control = context_[0x199 + a];
      if (control & ~0x33fu)
        throw std::runtime_error("unsupported graphics parameter interpolation");
      for (uint32_t c = 0; c < 4; ++c) {
        std::array<float, 3> values{};
        for (uint32_t k = 0; k < 3; ++k) {
          if (control & 32) {
            values[k] = ((control >> 8) & (c == 3 ? 1u : 2u)) ? 1.0f : 0.0f;
          } else {
            const auto address = addr_calc::rdna_buffer_address(
                ring[1], ring[3], ring_stride, true, indices[k], (control & 31) * 16 + c * 4, 0, 0);
            uint32_t bits = 0;
            if (memory.read_block_exact(ring_base + address.offset,
                                        {reinterpret_cast<uint8_t *>(&bits), sizeof(bits)},
                                        process_id) != AccessOutcome::Complete)
              throw std::runtime_error("graphics attribute read failed");
            values[k] = std::bit_cast<float>(bits);
          }
        }
        parameters[a * 12 + c * 3] = std::bit_cast<uint32_t>(values[0]);
        parameters[a * 12 + c * 3 + 1] = std::bit_cast<uint32_t>(values[1] - values[0]);
        parameters[a * 12 + c * 3 + 2] = std::bit_cast<uint32_t>(values[2] - values[0]);
      }
    }
    FragmentWave batch;
    batch.parameters = parameters;
    uint32_t used = 0;
    for (int y = min_y & ~1; y < max_y; y += 2) {
      for (int x = min_x & ~1; x < max_x; x += 2) {
        std::array<Fragment, 4> quad{};
        bool covered = false;
        for (int q = 0; q < 4; ++q) {
          auto &f = quad[q];
          f.x = x + (q & 1);
          f.y = y + (q >> 1);
          const double px = f.x + 0.5, py = f.y + 0.5;
          const double b1 = edge(v[2], v[0], px, py) / area;
          const double b2 = edge(v[0], v[1], px, py) / area;
          const double rw = (1 - b1 - b2) / v[0].w + b1 / v[1].w + b2 / v[2].w;
          f.i = b1 / v[1].w / rw;
          f.j = b2 / v[2].w / rw;
          bool inside = true;
          if (rectangle) {
            inside = px >= min_x && px < max_x && py >= min_y && py < max_y;
          } else {
            for (uint32_t k = 0; k < 3; ++k) {
              Point a = v[k], b = v[(k + 1) % 3];
              if (area < 0)
                std::swap(a, b);
              const double e = edge(a, b, px, py);
              const bool top_left = b.y < a.y || (b.y == a.y && b.x > a.x);
              inside &= e > 0 || (e == 0 && top_left);
            }
          }
          f.covered = inside && f.x >= left && f.x < right && f.y >= top && f.y < bottom;
          covered |= f.covered;
        }
        if (!covered)
          continue;
        std::copy(quad.begin(), quad.end(), batch.lanes.begin() + used);
        used += 4;
        if (used == fragment_wave_size_) {
          fragments_.push_back(std::move(batch));
          batch = FragmentWave{};
          batch.parameters = parameters;
          used = 0;
        }
      }
    }
    if (used)
      fragments_.push_back(std::move(batch));
  }
}

void GraphicsDraw::write_colors(GpuMemory &memory, uint32_t process_id) {
  for (const auto &batch : fragments_) {
    for (uint32_t lane = 0; lane < fragment_wave_size_; ++lane) {
      const auto &f = batch.lanes[lane];
      if (!f.covered || !f.mask)
        continue;
      std::array<uint32_t, 4> components = f.color;
      if (color_format_ == 4 || color_format_ == 7) {
        if (f.mask != 3)
          throw std::runtime_error("unsupported packed graphics color export mask");
        for (uint32_t c = 0; c < 4; ++c) {
          const uint16_t half = f.color[c / 2] >> (16 * (c % 2));
          components[c] =
              color_format_ == 7 ? half : std::bit_cast<uint32_t>(util::f16_to_f32(half));
        }
      } else if (f.mask != 15) {
        throw std::runtime_error("unsupported graphics color export mask");
      }
      std::array<uint8_t, 4> bytes{};
      pack_buffer_format(memory_format_, 0xfac, components, bytes);
      const auto address = gfx12_image_address(color_base_, f.x, f.y, width_, 4, swizzle_);
      if (!address || memory.write_block(*address, bytes, process_id) != AccessOutcome::Complete)
        throw std::runtime_error("graphics color write failed");
    }
  }
}

std::optional<DispatchEntry> GraphicsDraw::advance(GpuMemory &memory, uint32_t process_id) {
  if (fragment_stage_) {
    write_colors(memory, process_id);
    return next_vertex_group();
  }
  finish_vertices();
  rasterize(memory, process_id);
  fragment_stage_ = true;
  if (fragments_.empty())
    return next_vertex_group();
  return fragment_dispatch();
}

} // namespace rocjitsu::amdgpu
