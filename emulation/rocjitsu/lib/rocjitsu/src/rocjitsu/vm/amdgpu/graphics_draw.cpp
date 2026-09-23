// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/graphics_draw.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/image_address.h"
#include "rocjitsu/vm/amdgpu/image_metadata.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/raster_math.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace rocjitsu::amdgpu {
namespace {
constexpr uint32_t kTriangleList = 4, kTriangleStrip = 6, kRectangleList = 17;
// Factors 13-16 select the unsupported second-source export.
constexpr uint32_t kBlendZero = 0, kBlendOne = 1, kBlendSrcColor = 2, kBlendInvSrcColor = 3,
                   kBlendSrcAlpha = 4, kBlendInvSrcAlpha = 5, kBlendDstAlpha = 6,
                   kBlendInvDstAlpha = 7, kBlendDstColor = 8, kBlendInvDstColor = 9,
                   kBlendSrcAlphaSaturate = 10, kBlendConstantColor = 11,
                   kBlendInvConstantColor = 12, kBlendConstantAlpha = 17,
                   kBlendInvConstantAlpha = 18;
constexpr uint32_t kBlendAdd = 0, kBlendSubtract = 1, kBlendMin = 2, kBlendMax = 3,
                   kBlendReverseSubtract = 4;
constexpr uint32_t kBlendSeparateAlpha = 1u << 29, kBlendEnable = 1u << 30;
constexpr uint32_t kNumberUnorm = 0, kNumberUint = 4, kNumberSint = 5, kNumberSrgb = 6,
                   kNumberFloat = 7;
constexpr uint32_t kBufR32Uint = 20, kBufRgba8Unorm = 42, kBufRg32Uint = 48, kBufRgba16Unorm = 51,
                   kBufRgba16Float = 57, kBufRgba32Uint = 61, kBufRgba32Float = 63;

constexpr uint32_t kExport32R = 1, kExport32Gr = 2, kExportFp16Abgr = 4, kExportUnorm16Abgr = 5,
                   kExportSnorm16Abgr = 6, kExportUint16Abgr = 7, kExportSint16Abgr = 8,
                   kExport32Abgr = 9;

// CB color formats use separate data and number fields; pack_buffer_format uses
// the combined GFX11 buffer format table.
uint32_t color_buffer_format(uint32_t data_format, uint32_t number_format) {
  switch (data_format) {
  case 10: // COLOR_8_8_8_8
    if (number_format == kNumberSrgb)
      return kBufRgba8Unorm; // Encode RGB separately; alpha remains linear UNORM.
    if (number_format == kNumberUnorm || number_format == kNumberUint ||
        number_format == kNumberSint)
      return kBufRgba8Unorm + number_format;
    break;
  case 12: // COLOR_16_16_16_16
    if (number_format <= kNumberSint || number_format == kNumberFloat)
      // FLOAT follows SINT in the buffer table, skipping NUMBER_SRGB.
      return kBufRgba16Unorm + (number_format == kNumberFloat ? kNumberSint + 1 : number_format);
    break;
  case 4:  // COLOR_32
  case 11: // COLOR_32_32
  case 14: // COLOR_32_32_32_32
    if (number_format == kNumberUint || number_format == kNumberSint ||
        number_format == kNumberFloat) {
      const uint32_t base = data_format == 4    ? kBufR32Uint
                            : data_format == 11 ? kBufRg32Uint
                                                : kBufRgba32Uint;
      return base + (number_format == kNumberFloat ? kNumberSint - kNumberUint + 1
                                                   : number_format - kNumberUint);
    }
    break;
  }
  return 0;
}

bool supported_export_format(uint32_t format, uint32_t components) {
  switch (format) {
  case kExport32R:
    return components == 1;
  case kExport32Gr:
    return components == 2;
  case kExportFp16Abgr:
  case kExportUnorm16Abgr:
  case kExportSnorm16Abgr:
  case kExportUint16Abgr:
  case kExportSint16Abgr:
  case kExport32Abgr:
    return true;
  default:
    return false;
  }
}

bool supported_blend(uint32_t control) {
  const uint32_t operation = (control >> 5) & 7;
  const auto supported_factor = [](uint32_t value) {
    return value <= kBlendInvConstantColor || value == kBlendConstantAlpha ||
           value == kBlendInvConstantAlpha;
  };
  return operation <= kBlendReverseSubtract &&
         (operation == kBlendMin || operation == kBlendMax ||
          (supported_factor(control & 31) && supported_factor((control >> 8) & 31)));
}

template <typename T>
T blend_factor(uint32_t factor, uint32_t component, const std::array<T, 4> &source,
               const std::array<T, 4> &destination, const std::array<T, 4> &constant) {
  switch (factor) {
  case kBlendZero:
    return 0;
  case kBlendOne:
    return 1;
  case kBlendSrcColor:
    return source[component];
  case kBlendInvSrcColor:
    return 1 - source[component];
  case kBlendSrcAlpha:
    return source[3];
  case kBlendInvSrcAlpha:
    return 1 - source[3];
  case kBlendDstAlpha:
    return destination[3];
  case kBlendInvDstAlpha:
    return 1 - destination[3];
  case kBlendDstColor:
    return destination[component];
  case kBlendInvDstColor:
    return 1 - destination[component];
  case kBlendSrcAlphaSaturate:
    return component == 3 ? 1 : std::min(source[3], 1 - destination[3]);
  case kBlendConstantColor:
    return constant[component];
  case kBlendInvConstantColor:
    return 1 - constant[component];
  case kBlendConstantAlpha:
    return constant[3];
  case kBlendInvConstantAlpha:
    return 1 - constant[3];
  default:
    throw std::runtime_error("unsupported graphics blend factor");
  }
}

template <typename T>
T blend_component(uint32_t control, uint32_t component, const std::array<T, 4> &source,
                  const std::array<T, 4> &destination, const std::array<T, 4> &constant) {
  const uint32_t operation = (control >> 5) & 7;
  if (operation == kBlendMin)
    return std::min(source[component], destination[component]);
  if (operation == kBlendMax)
    return std::max(source[component], destination[component]);
  const T source_term =
      source[component] * blend_factor(control & 31, component, source, destination, constant);
  const T destination_term = destination[component] * blend_factor((control >> 8) & 31, component,
                                                                   source, destination, constant);
  if (operation == kBlendSubtract)
    return source_term - destination_term;
  if (operation == kBlendReverseSubtract)
    return destination_term - source_term;
  if (operation == kBlendAdd)
    return source_term + destination_term;
  throw std::runtime_error("unsupported graphics blend operation");
}

// Color-buffer quantization thresholds for FP16 sRGB exports. Each entry is
// the first positive half encoding producing the next byte value. Captures of
// all 65536 FP16 inputs agree on RDNA3 and RDNA4; applying the ideal sRGB curve
// instead changes values near these boundaries.
uint8_t srgb_color_byte(float value) {
  if (!(value > 0))
    return 0;
  if (value >= 1)
    return 255;
  static constexpr uint16_t boundaries[] = {
      0x0900, 0x0f80, 0x1200, 0x1440, 0x1580, 0x16c0, 0x1800, 0x18a0, 0x1940, 0x19e0, 0x1a80,
      0x1b30, 0x1be0, 0x1c50, 0x1cb0, 0x1d20, 0x1d80, 0x1e00, 0x1e70, 0x1ee0, 0x1f60, 0x1ff0,
      0x2040, 0x2088, 0x20d0, 0x2120, 0x2170, 0x21c0, 0x2220, 0x2270, 0x22d0, 0x2330, 0x2390,
      0x2400, 0x2430, 0x2468, 0x24a0, 0x24d8, 0x2510, 0x2550, 0x2590, 0x25d0, 0x2610, 0x2650,
      0x2690, 0x26e0, 0x2720, 0x2770, 0x27b8, 0x2800, 0x2828, 0x2850, 0x2878, 0x28a0, 0x28d0,
      0x28f8, 0x2928, 0x2950, 0x2980, 0x29b0, 0x29e0, 0x2a10, 0x2a40, 0x2a78, 0x2aa8, 0x2ae0,
      0x2b10, 0x2b48, 0x2b80, 0x2bb8, 0x2bf0, 0x2c18, 0x2c34, 0x2c54, 0x2c70, 0x2c90, 0x2cb0,
      0x2cd0, 0x2cf0, 0x2d10, 0x2d34, 0x2d58, 0x2d78, 0x2d98, 0x2dc0, 0x2de0, 0x2e08, 0x2e2c,
      0x2e50, 0x2e78, 0x2ea0, 0x2ec8, 0x2ef0, 0x2f18, 0x2f40, 0x2f68, 0x2f90, 0x2fbc, 0x2fe8,
      0x3008, 0x3020, 0x3034, 0x304c, 0x3064, 0x3078, 0x3090, 0x30a8, 0x30c0, 0x30d8, 0x30f0,
      0x3108, 0x3124, 0x313c, 0x3154, 0x3170, 0x3188, 0x31a4, 0x31c0, 0x31d8, 0x31f4, 0x3210,
      0x322c, 0x3248, 0x3264, 0x3280, 0x32a0, 0x32bc, 0x32d8, 0x32f8, 0x3318, 0x3334, 0x3354,
      0x3370, 0x3390, 0x33b0, 0x33d0, 0x33f0, 0x3408, 0x3418, 0x342a, 0x343c, 0x344c, 0x345c,
      0x346e, 0x3480, 0x3490, 0x34a2, 0x34b4, 0x34c6, 0x34d8, 0x34ea, 0x34fc, 0x3510, 0x3522,
      0x3534, 0x3548, 0x355c, 0x3570, 0x3582, 0x3596, 0x35aa, 0x35be, 0x35d2, 0x35e8, 0x35fc,
      0x3610, 0x3624, 0x3638, 0x3650, 0x3664, 0x3678, 0x3690, 0x36a4, 0x36bc, 0x36d0, 0x36e8,
      0x36fc, 0x3714, 0x372c, 0x3740, 0x3758, 0x3770, 0x3788, 0x37a0, 0x37b8, 0x37d0, 0x37e8,
      0x3800, 0x380c, 0x3818, 0x3824, 0x3832, 0x383e, 0x384a, 0x3858, 0x3864, 0x3870, 0x387e,
      0x388c, 0x3898, 0x38a6, 0x38b4, 0x38c0, 0x38ce, 0x38dc, 0x38e8, 0x38f8, 0x3904, 0x3914,
      0x3920, 0x3930, 0x393c, 0x394c, 0x395a, 0x3968, 0x3978, 0x3986, 0x3994, 0x39a4, 0x39b2,
      0x39c0, 0x39d0, 0x39e0, 0x39f0, 0x39fe, 0x3a0e, 0x3a1c, 0x3a2c, 0x3a3c, 0x3a4c, 0x3a5c,
      0x3a6c, 0x3a7c, 0x3a8c, 0x3a9c, 0x3aae, 0x3abe, 0x3ad0, 0x3ae0, 0x3af0, 0x3b00, 0x3b12,
      0x3b24, 0x3b34, 0x3b44, 0x3b58, 0x3b68, 0x3b7a, 0x3b8c, 0x3b9c, 0x3bb0, 0x3bc0, 0x3bd4,
      0x3be4, 0x3bf8,
  };
  const uint16_t half = util::f32_to_f16(value);
  return static_cast<uint8_t>(std::upper_bound(std::begin(boundaries), std::end(boundaries), half) -
                              std::begin(boundaries));
}
} // namespace

GraphicsDraw::GraphicsDraw(const Pm4QueueState &state, rj_code_arch_t arch, uint32_t vertices,
                           std::vector<uint32_t> indices)
    : arch_(arch), vertex_count_(vertices), total_vertices_(vertices),
      instance_count_(state.num_instances), primitive_type_(state.uconfig_registers[0x242]),
      indices_(std::move(indices)), sh_(state.sh_registers), context_(state.context_registers) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch != ROCJITSU_CODE_ARCH_RDNA4)
    throw std::runtime_error("graphics draw requires RDNA3 or RDNA4");
  if (!instance_count_ || vertices < 3 || vertices > (1u << 20) ||
      (primitive_type_ != kTriangleList && primitive_type_ != kTriangleStrip &&
       primitive_type_ != kRectangleList))
    throw std::runtime_error("unsupported graphics primitive, vertex count, or instance count");
  const auto &ctx = state.context_registers;
  for (uint32_t target = 0; target < colors_.size(); ++target) {
    auto &color = colors_[target];
    const uint32_t dst = 0x318 + 9 * target;
    if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
      color.max_mip = (ctx[dst + 7] >> 19) & 31;
      color.mip = ctx[dst + 2] & 31;
      continue;
    }
    // GFX11 color blocks have fifteen registers; GFX12 blocks have nine.
    // Always read the original snapshot because the two layouts overlap.
    const uint32_t src = 0x318 + 15 * target;
    if (ctx[src + 6] & (1u << 22))
      color.metadata = addr_calc::buffer_virtual_address(
          ((uint64_t{ctx[0x3a8 + target] & 255} << 32) | ctx[src + 13]) << 8);
    context_[dst] = ctx[src];
    context_[dst + 1] = ctx[src + 3];
    context_[dst + 2] = 0;
    context_[dst + 3] = ctx[src + 5];
    const uint32_t attrib2 = ctx[0x3b0 + target];
    context_[dst + 6] = (attrib2 & 0x3fff) | (((attrib2 >> 14) & 0x3fff) << 16);
    context_[dst + 7] = ctx[0x3b8 + target];
    context_[0x3b0 + target] = ctx[src + 4];
    color.max_mip = attrib2 >> 28;
    color.mip = (ctx[src + 3] >> 26) & 15;
  }
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    // Normalize relocated registers into the GFX12 slots used below. Read the
    // original snapshot because several source and destination slots overlap.
    if ((ctx[0x200] & 2) && (ctx[0x10] & (1u << 29))) {
      if ((ctx[0x11] & 1) || !(ctx[0x2af] & (1u << 18)))
        throw std::runtime_error("unsupported GFX11 stencil or unaligned HTILE surface");
      depth_metadata_ =
          addr_calc::buffer_virtual_address(((uint64_t{ctx[0x1e] & 255} << 32) | ctx[5]) << 8);
      depth_clear_ = ctx[0xb];
    }
    for (const auto [dst, src] : {std::pair{0x1b, 0x203},
                                  {0x1c, 0x200},
                                  {0x190, 0x1b6},
                                  {0x195, 0x1c5},
                                  {0x197, 0x1b3},
                                  {0x198, 0x1b4},
                                  {0x115, 0xb4},
                                  {0x116, 0xb5},
                                  {0x205, 0x206},
                                  {0x207, 0x205},
                                  {0x206, 0x207},
                                  {0x214, 0x8e},
                                  {0x215, 0x8f},
                                  {0x216, 0x202}})
      context_[dst] = ctx[src];
    context_[0x19] = (ctx[3] >> 16) & 1; // DISABLE_VIEWPORT_CLAMP
    for (uint32_t i = 0; i < 32; ++i)
      context_[0x199 + i] = ctx[0x191 + i];
    sh_[0x31] = ((ctx[0x1b1] >> 1) & 31) | ((ctx[0x1b6] & 63) << 11);
    sh_[0x84] = state.sh_registers[0x88];
    sh_[0x85] = state.sh_registers[0x89];
    context_[5] = (ctx[7] & 0x3fff) | (((ctx[7] >> 16) & 0x3fff) << 16);
    context_[6] = ctx[0x10];
    context_[8] = ctx[0x12];
    context_[9] = ctx[0x1a];
    context_[10] = ctx[0x14];
    context_[11] = ctx[0x1c];
    context_[1] = ctx[2] & 0xc0ffffff;
    context_[2] = ctx[2] & 0x3f000000;
  }
  select_vertex_group();
  if (!indices_.empty() && indices_.size() != total_vertices_)
    throw std::runtime_error("graphics index count does not match draw");
  if (!indices_.empty() && (state.uconfig_registers[0x24b] & 1))
    throw std::runtime_error("graphics primitive restart is not implemented");
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
      // INPUT_ADDR reserves the VGPRs in architectural order, including
      // inputs whose calculation is disabled in INPUT_ENA.
      uint32_t vgpr = 0;
      const auto put = [&](float value) {
        wave.debug_write_vgpr(vgpr++, lane, std::bit_cast<uint32_t>(value));
      };
      for (uint32_t input = 0; input < 7; ++input) {
        if (!(context_[0x198] & (1u << input)))
          continue;
        if (input == 3) {
          for (float value : f.pull_model)
            put(value);
        } else if (input < 3) {
          // With one sample, center, sample and covered centroid coincide.
          put(f.i);
          put(f.j);
        } else {
          put(f.linear_i);
          put(f.linear_j);
        }
      }
      // POS_W_FLOAT supplies W; PERSP_PULL_MODEL above supplies 1/W.
      const std::array<float, 4> position{float(f.x) + 0.5f, float(f.y) + 0.5f, f.z,
                                          1.0f / f.pull_model[2]};
      for (uint32_t component = 0; component < 4; ++component)
        if (context_[0x198] & (1u << (8 + component)))
          put(position[component]);
      if (context_[0x198] & (1u << 15))
        wave.debug_write_vgpr(vgpr++, lane, (uint32_t(f.x) & 0xffff) | (uint32_t(f.y) << 16));
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
    const uint32_t index = indices_.empty()       ? first_vertex_ + lane
                           : lane < vertex_count_ ? indices_[first_vertex_ + lane]
                                                  : 0;
    wave.debug_write_vgpr(gfx12 ? 3 : 5, lane, index);
    const uint32_t components = (sh_[0x8b] >> 16) & 3;
    if (components >= (gfx12 ? 1u : 3u))
      wave.debug_write_vgpr(gfx12 ? 4 : 8, lane, instance_);
  }
}

void GraphicsDraw::export_mask(Wavefront &wave, uint64_t mask) {
  if (!fragment_stage_)
    return;
  auto &batch = fragments_.at(wave.wg_coord()[0]);
  for (uint32_t lane = 0; lane < fragment_wave_size_; ++lane)
    batch.lanes[lane].covered &= (mask & (uint64_t{1} << lane)) != 0;
}

void GraphicsDraw::export_lane(Wavefront &wave, uint32_t lane, uint32_t target, uint32_t mask,
                               const std::array<uint32_t, 4> &values) {
  if (fragment_stage_) {
    if (target == 8 && !(mask & ~1u)) {
      if (mask)
        fragments_.at(wave.wg_coord()[0]).lanes[lane].z = std::bit_cast<float>(values[0]);
      return;
    }
    if (target >= colors_.size()) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return;
    }
    auto &exported = fragments_.at(wave.wg_coord()[0]).lanes[lane].exports[target];
    for (uint32_t i = 0; i < 4; ++i)
      if (mask & (1u << i))
        exported.values[i] = values[i];
    exported.mask |= mask;
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
    if ((context_[0x206] & (1u << 19)) && (layer_viewport_[i] >> 16))
      throw std::runtime_error("graphics viewport selection is not implemented");
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
  const uint32_t rsrc1 = sh_[0xa];
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

void GraphicsDraw::prepare_colors() {
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t color_control = context_[0x216];
  const uint32_t color_mode = (color_control >> 4) & 7;
  color_enabled_ = (context_[0x214] & context_[0x215]) != 0 && color_mode != 0;
  if (color_enabled_ &&
      (color_mode != 1 || ((color_control >> 16) & 255) != 0xcc || (color_control & 8)))
    throw std::runtime_error("unsupported graphics color mode, logic operation, or degamma");
  width_ = height_ = 0;
  uint32_t export_index = 0;
  for (uint32_t target = 0; target < colors_.size(); ++target) {
    auto &color = colors_[target];
    const uint32_t shader_mask = (context_[0x215] >> (4 * target)) & 15;
    // Shader exports and their format nibbles omit holes in CB_SHADER_MASK.
    // CB_TARGET_MASK can disable writes without removing a shader export slot.
    color.export_index = export_index;
    color.export_format = (context_[0x195] >> (4 * export_index)) & 15;
    export_index += shader_mask != 0;
    color.write_mask = color_enabled_ ? (context_[0x214] >> (4 * target)) & shader_mask : 0;
    if (!color.write_mask)
      continue;
    const uint32_t block = 0x318 + 9 * target;
    const uint32_t info = context_[0x3b0 + target];
    const uint32_t attrib = context_[block + 3], attrib2 = context_[block + 6],
                   attrib3 = context_[block + 7];
    util::Logger::cp("graphics target ", target, " info=", std::hex, info, " attrib=", attrib, ",",
                     attrib2, ",", attrib3, " export=", color.export_format,
                     " mask=", color.write_mask, std::dec);
    const uint32_t data_format = info & 31, number_format = (info >> 8) & 7;
    color.srgb = number_format == kNumberSrgb;
    color.memory_format = color_buffer_format(data_format, number_format);
    color.bytes = buffer_format_bytes(color.memory_format);
    color.components = buffer_format_components(color.memory_format);
    color.blend = context_[0x1e0 + target];
    uint32_t allowed_attrib = gfx12 ? 0u : 0x30u;
    // FORCE_DST_ALPHA_1 has no effect on these non-blended targets without alpha.
    if (color.components < 4 && !(color.blend & kBlendEnable))
      allowed_attrib |= 1u << 2;
    const uint32_t layer_bits = gfx12 ? 14 : 13, layer_mask = (1u << layer_bits) - 1;
    const uint32_t view = context_[block + 1];
    color.first_layer = view & layer_mask;
    color.last_layer = (view >> layer_bits) & layer_mask;
    if (!color.memory_format || (color.srgb && color.export_format != kExportFp16Abgr) ||
        (info & (3u << 11)) || (attrib & ~allowed_attrib) || ((attrib3 >> 24) & 3) > 1 ||
        (attrib3 & (1u << layer_bits)) || (context_[block + 2] & ~31u) ||
        color.first_layer > color.last_layer || color.last_layer > (attrib3 & layer_mask) ||
        (view & (gfx12 ? 0xf0000000u : 0xc0000000u)) ||
        !supported_export_format(color.export_format, color.components))
      throw std::runtime_error("unsupported graphics color state");
    if ((color.blend & kBlendEnable) &&
        (color.srgb ||
         (color.memory_format != kBufRgba8Unorm && color.memory_format != kBufRgba16Float &&
          color.memory_format != kBufRgba32Float) ||
         !supported_blend(color.blend) ||
         ((color.blend & kBlendSeparateAlpha) && !supported_blend(color.blend >> 16))))
      throw std::runtime_error("unsupported graphics blend operation or integer attachment");
    color.swizzle = gfx12 ? (attrib3 >> 15) & 7 : (attrib3 >> 14) & 31;
    color.pipe_aligned = attrib3 & (1u << 30);
    const auto mip = image_mip_layout(gfx12, color.swizzle, color.bytes, (attrib2 >> 16) + 1,
                                      (attrib2 & 0xffff) + 1, color.max_mip + 1, color.mip);
    if (!mip || (color.metadata && color.max_mip))
      throw std::runtime_error("unsupported graphics color mip layout");
    color.width = mip->width;
    color.height = mip->height;
    if (width_ && (width_ != color.width || height_ != color.height))
      throw std::runtime_error("graphics color attachment extents must match");
    width_ = color.width;
    height_ = color.height;
    color.pitch = mip->pitch;
    color.slice_size = mip->slice_size;
    color.tail_x = mip->tail_x;
    color.tail_y = mip->tail_y;
    color.base = addr_calc::buffer_virtual_address(
                     ((uint64_t{context_[0x390 + target] & 255} << 32) | context_[block]) << 8) +
                 mip->offset;
  }
}

void GraphicsDraw::rasterize(const GpuVmAccess &memory) {
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t clip_control = context_[0x204];
  if (clip_control & (1u << 22)) // DX_RASTERIZATION_KILL
    return;
  if (clip_control & (0x3fu | (1u << 28)))
    throw std::runtime_error("graphics user clip planes are not implemented");
  if (sh_[0xb] & 1) // SPI_SHADER_PGM_RSRC2_PS.SCRATCH_EN
    throw std::runtime_error("fragment scratch is not implemented");
  if (context_[0x1b] & (1u << 12)) // DB_SHADER_CONTROL.DEPTH_BEFORE_SHADER
    throw std::runtime_error("graphics early fragment tests are not implemented");
  const uint32_t polygon_mode = (context_[0x207] >> 3) & 3;
  if (polygon_mode && (polygon_mode != 1 || ((context_[0x207] >> 5) & 63) != (2 | (2 << 3))))
    throw std::runtime_error("graphics point or line polygon modes are not implemented");
  if ((context_[0x207] & (7u << 11)) &&
      ((context_[0x2e0] | context_[0x2e1] | context_[0x2e2] | context_[0x2e3]) & 0x7fffffffu))
    throw std::runtime_error("graphics polygon depth bias is not implemented");
  if (context_[0x2f8] & 7)
    throw std::runtime_error("graphics multisample rasterization is not implemented");
  if (context_[0x2f9] != 0x2d)
    throw std::runtime_error("unsupported graphics pixel center or subpixel rounding");
  prepare_colors();
  depth_control_ = context_[0x1c];
  if (depth_control_ & ~0x76u)
    throw std::runtime_error("unsupported graphics stencil or depth bounds test");
  if ((context_[0x198] & ~0x8f7fu) || (context_[0x197] & ~context_[0x198]))
    throw std::runtime_error("unsupported graphics fragment inputs");

  if (depth_control_ & 2) {
    const uint32_t zinfo = context_[6];
    depth_width_ = (context_[5] & 0xffff) + 1;
    depth_height_ = (context_[5] >> 16) + 1;
    depth_swizzle_ = (zinfo >> 4) & 31;
    depth_bytes_ = (zinfo & 3) == 1 ? 2 : 4;
    depth_base_ =
        addr_calc::buffer_virtual_address(((uint64_t{context_[9] & 255} << 32) | context_[8]) << 8);
    const uint64_t write_base = addr_calc::buffer_virtual_address(
        ((uint64_t{context_[11] & 255} << 32) | context_[10]) << 8);
    util::Logger::cp("graphics depth info=", std::hex, zinfo, " view=", context_[1], ",",
                     context_[2], " base=", depth_base_, " write=", write_base, std::dec);
    if (((zinfo & 15) != 3 && (zinfo & 15) != 1) || (zinfo & (31u << 15)) ||
        !(gfx12 ? gfx12_image_offset(0, 0, depth_width_, depth_bytes_, depth_swizzle_)
                : gfx11_image_offset(0, 0, depth_width_, depth_bytes_, depth_swizzle_)) ||
        context_[1] || (context_[2] & 0x7c000000u) ||
        ((depth_control_ & 4) && ((context_[2] & (1u << 24)) || depth_base_ != write_base)))
      throw std::runtime_error("unsupported graphics depth surface");
    if (!color_enabled_) {
      width_ = depth_width_;
      height_ = depth_height_;
    } else if (depth_width_ != width_ || depth_height_ != height_) {
      throw std::runtime_error("graphics depth and color extents must match");
    }
  } else if (!color_enabled_) {
    throw std::runtime_error("graphics draw has no supported attachment");
  }
  if (width_ > 4096 || height_ > 4096)
    throw std::runtime_error("graphics render target exceeds supported dimensions");
  fragment_wave_size_ = (context_[0x190] & (1u << 15)) ? 32 : 64;
  const uint32_t attributes = (sh_[0x31] >> 11) & 63;
  if (attributes > 32 || (sh_[0x31] >> 17))
    throw std::runtime_error("unsupported graphics parameter count");
  std::array<uint32_t, 4> ring{};
  if (attributes) {
    const uint64_t table = (uint64_t{sh_[0x85]} << 32) | sh_[0x84];
    if (memory.read(table + 0xa0, {reinterpret_cast<std::byte *>(ring.data()), sizeof(ring)}) !=
        VmAccessOutcome::Complete)
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
  const uint32_t scissor_mask = gfx12 ? 0xffff : 0x7fff;
  int left = std::max(0, int(context_[0x90] & scissor_mask));
  int top = std::max(0, int((context_[0x90] >> 16) & scissor_mask));
  // GFX12 changed the bottom-right scissor bounds from exclusive to inclusive.
  int right = std::min(int(width_), int(context_[0x91] & scissor_mask) + int(gfx12));
  int bottom = std::min(int(height_), int((context_[0x91] >> 16) & scissor_mask) + int(gfx12));
  if (context_[0x292] & 2) { // PA_SC_MODE_CNTL_0.VPORT_SCISSOR_ENABLE
    const bool window_offset = !gfx12 && !(context_[0x94] & (1u << 31));
    const int x_offset = window_offset ? int16_t(context_[0x80]) : 0;
    const int y_offset = window_offset ? int16_t(context_[0x80] >> 16) : 0;
    left = std::max(left, int(context_[0x94] & scissor_mask) + x_offset);
    top = std::max(top, int((context_[0x94] >> 16) & scissor_mask) + y_offset);
    right = std::min(right, int(context_[0x95] & scissor_mask) + x_offset + int(gfx12));
    bottom = std::min(bottom, int((context_[0x95] >> 16) & scissor_mask) + y_offset + int(gfx12));
  }
  if (left >= right || top >= bottom)
    return;
  for (uint32_t p = 0; p < primitive_count(); ++p) {
    if (primitives_[p] & (1u << 31))
      continue;
    std::array<uint32_t, 3> indices{};
    struct Point {
      double x, y, w, z;
      std::array<double, 3> barycentric{};
    };
    std::array<Point, 3> v{}, screen{}, clip_positions{};
    uint32_t outside_near = 0, outside_far = 0, nonpositive_w = 0;
    bool clip_xy = false;
    for (uint32_t k = 0; k < 3; ++k) {
      const uint32_t index_bits = gfx12 ? 9 : 10;
      indices[k] = (primitives_[p] >> (index_bits * k)) & ((1u << index_bits) - 1);
      if (indices[k] >= vertex_count_)
        throw std::runtime_error("graphics primitive index exceeds vertex exports");
      const auto &position = positions_[indices[k]];
      const float w = std::bit_cast<float>(position[3]);
      const float x = std::bit_cast<float>(position[0]);
      const float y = std::bit_cast<float>(position[1]);
      if (!std::isfinite(w) || !std::isfinite(x) || !std::isfinite(y)) {
        throw std::runtime_error("graphics homogeneous clipping is not implemented");
      }
      const float z = std::bit_cast<float>(position[2]);
      if (!std::isfinite(z))
        throw std::runtime_error("graphics nonfinite depth is not implemented");
      nonpositive_w += w <= 0;
      clip_positions[k] = {x, y, w, z};
      clip_positions[k].barycentric[k] = 1;
      if (!(clip_control & (1u << 16))) {
        const float near = (clip_control & (1u << 19)) ? 0.0f : -w;
        outside_near += !(clip_control & (1u << 26)) && z < near;
        outside_far += !(clip_control & (1u << 27)) && z > w;
      }
      v[k] = w == 0 ? Point{0, 0, w, 0}
                    : Point{raster::viewport_coordinate(x, w, sx, ox),
                            raster::viewport_coordinate(y, w, sy, oy), w, z / w};
      screen[k] = v[k];
      // Finite homogeneous inputs can overflow the initial FP32 projection.
      // Let guard-band clipping produce finite coordinates before validation.
      clip_xy |= w <= 0 || std::abs(v[k].x) > (1 << 20) || std::abs(v[k].y) > (1 << 20);
    }
    // Layer selection uses the provoking vertex before interpolation reorders vertices.
    const uint32_t provoking = (context_[0x207] & (1u << 19)) ? 2 : 0;
    const uint32_t relative_layer =
        (context_[0x206] & (1u << 18)) ? layer_viewport_[indices[provoking]] & 0xffff : 0;
    if (color_enabled_ && std::none_of(colors_.begin(), colors_.end(), [&](const auto &color) {
          return color.write_mask && relative_layer <= color.last_layer - color.first_layer;
        }))
      continue;
    // Depth still has a single slice; color attachments have independent array views.
    if ((depth_control_ & 2) && relative_layer)
      throw std::runtime_error("graphics layered depth attachments are not implemented");
    if (nonpositive_w == 3 || outside_near == 3 || outside_far == 3)
      continue;
    // Clip coverage in homogeneous coordinates and carry the original
    // barycentrics so added vertices do not replace shader-visible parameters.
    std::vector<Point> coverage(v.begin(), v.end());
    if (clip_xy || outside_near || outside_far) {
      if (primitive_type_ == kRectangleList)
        throw std::runtime_error("graphics clipped rectangles are not implemented");
      coverage.assign(clip_positions.begin(), clip_positions.end());
      const auto clip = [&](const auto &distance) {
        std::vector<Point> output;
        if (coverage.empty())
          return;
        Point previous = coverage.back();
        double previous_distance = distance(previous);
        for (const Point &current : coverage) {
          const double current_distance = distance(current);
          if ((previous_distance < 0 && current_distance > 0) ||
              (previous_distance > 0 && current_distance < 0)) {
            const double t = previous_distance / (previous_distance - current_distance);
            output.push_back({std::lerp(previous.x, current.x, t),
                              std::lerp(previous.y, current.y, t),
                              std::lerp(previous.w, current.w, t),
                              std::lerp(previous.z, current.z, t),
                              {std::lerp(previous.barycentric[0], current.barycentric[0], t),
                               std::lerp(previous.barycentric[1], current.barycentric[1], t),
                               std::lerp(previous.barycentric[2], current.barycentric[2], t)}});
          }
          if (current_distance >= 0)
            output.push_back(current);
          previous = current;
          previous_distance = current_distance;
        }
        coverage = std::move(output);
      };
      if (outside_near)
        clip([&](Point point) { return point.z + ((clip_control & (1u << 19)) ? 0 : point.w); });
      if (outside_far)
        clip([](Point point) { return point.w - point.z; });
      if (clip_xy) {
        const double gx = std::bit_cast<float>(context_[gfx12 ? 0x10d : 0x2fc]);
        const double gy = std::bit_cast<float>(context_[gfx12 ? 0x10b : 0x2fa]);
        if (!std::isfinite(gx) || !std::isfinite(gy) || gx < 1 || gy < 1)
          throw std::runtime_error("unsupported graphics guard band");
        clip([&](Point point) { return point.w * gx + point.x; });
        clip([&](Point point) { return point.w * gx - point.x; });
        clip([&](Point point) { return point.w * gy + point.y; });
        clip([&](Point point) { return point.w * gy - point.y; });
      }
      if (coverage.size() < 3)
        continue;
      // A clipped vertex at the homogeneous origin has no projected area.
      std::erase_if(coverage, [](Point point) { return point.w == 0; });
      if (coverage.size() < 3)
        continue;
      for (auto &point : coverage) {
        point.x = raster::viewport_coordinate(point.x, point.w, sx, ox);
        point.y = raster::viewport_coordinate(point.y, point.w, sy, oy);
        point.z /= point.w;
      }
      const auto same_position = [](Point a, Point b) { return a.x == b.x && a.y == b.y; };
      coverage.erase(std::unique(coverage.begin(), coverage.end(), same_position), coverage.end());
      if (coverage.size() > 1 && same_position(coverage.front(), coverage.back()))
        coverage.pop_back();
      if (coverage.size() < 3)
        continue;
    }
    for (const auto &point : coverage)
      if (!std::isfinite(point.x) || !std::isfinite(point.y))
        throw std::runtime_error("graphics clipped vertex has nonfinite raster coordinates");
    const auto edge = [](Point a, Point b, double x, double y) {
      return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
    };
    double area = 0;
    for (uint32_t k = 1; k + 1 < coverage.size(); ++k)
      area += edge(coverage[0], coverage[k], coverage[k + 1].x, coverage[k + 1].y);
    if (area == 0)
      continue;
    const bool front = (area < 0) != bool(context_[0x207] & 4);
    if ((front && (context_[0x207] & 1)) || (!front && (context_[0x207] & 2)))
      continue;
    // Choose the smallest W, then the leftmost vertex. Equal-W axis-aligned
    // right triangles use the corner joining the two axis-aligned edges.
    uint32_t origin = 0;
    for (uint32_t k = 1; k < 3; ++k)
      if (screen[k].w < screen[origin].w ||
          (screen[k].w == screen[origin].w &&
           (screen[k].x < screen[origin].x ||
            (screen[k].x == screen[origin].x && screen[k].y < screen[origin].y))))
        origin = k;
    for (uint32_t k = 0; k < 3; ++k) {
      const auto &a = screen[k], &b = screen[(k + 1) % 3], &c = screen[(k + 2) % 3];
      if (a.w == b.w && a.w == c.w && ((a.x == b.x && a.y == c.y) || (a.y == b.y && a.x == c.x)))
        origin = k;
    }
    if (clip_xy) {
      // Establish planes from finite clipped coordinates instead of projecting
      // vertices arbitrarily close to W=0. Keep barycentrics in the original
      // triangle so explicit interpolation and provoking vertices stay intact.
      origin = 0;
      uint32_t first = 0;
      for (uint32_t k = 1; k < coverage.size(); ++k)
        if (coverage[k].w < coverage[first].w ||
            (coverage[k].w == coverage[first].w && coverage[k].x < coverage[first].x))
          first = k;
      screen[0] = coverage[first];
      double largest = 0;
      for (uint32_t k = 1; k + 1 < coverage.size(); ++k) {
        const auto &a = coverage[(first + k) % coverage.size()];
        const auto &b = coverage[(first + k + 1) % coverage.size()];
        const double candidate = std::abs(edge(screen[0], a, b.x, b.y));
        if (candidate > largest) {
          largest = candidate;
          screen[1] = a;
          screen[2] = b;
        }
      }
      if (largest == 0)
        continue;
    }
    std::rotate(indices.begin(), indices.begin() + origin, indices.end());
    std::rotate(v.begin(), v.begin() + origin, v.end());
    std::rotate(screen.begin(), screen.begin() + origin, screen.end());
    const double interpolation_area = edge(screen[0], screen[1], screen[2].x, screen[2].y);
    const float inverse_area = raster::truncate_float(1.0 / interpolation_area);
    const auto plane = [&](float a, float b, float c) {
      return raster::Plane{
          raster::plane_gradient(screen[2].y - screen[0].y, screen[0].y - screen[1].y,
                                 double(b) - a, double(c) - a, inverse_area),
          raster::plane_gradient(screen[0].x - screen[2].x, screen[1].x - screen[0].x,
                                 double(b) - a, double(c) - a, inverse_area),
          a};
    };
    const auto original_weight = [&](uint32_t vertex, uint32_t component) {
      return clip_xy ? screen[vertex].barycentric[component] : double(vertex == component);
    };
    const auto perspective_plane = [&](uint32_t component) {
      return plane(original_weight(0, component) / screen[0].w,
                   original_weight(1, component) / screen[1].w,
                   original_weight(2, component) / screen[2].w);
    };
    const raster::Plane plane_iw = perspective_plane(1);
    const raster::Plane plane_jw = perspective_plane(2);
    const raster::Plane plane_rw = plane(1.0 / screen[0].w, 1.0 / screen[1].w, 1.0 / screen[2].w);
    const raster::Plane plane_i = clip_xy ? plane(original_weight(0, 1) * v[1].w / screen[0].w,
                                                  original_weight(1, 1) * v[1].w / screen[1].w,
                                                  original_weight(2, 1) * v[1].w / screen[2].w)
                                          : plane(0, 1, 0);
    const raster::Plane plane_j = clip_xy ? plane(original_weight(0, 2) * v[2].w / screen[0].w,
                                                  original_weight(1, 2) * v[2].w / screen[1].w,
                                                  original_weight(2, 2) * v[2].w / screen[2].w)
                                          : plane(0, 0, 1);
    const raster::Plane plane_z = plane(screen[0].z, screen[1].z, screen[2].z);
    const bool rectangle = primitive_type_ == kRectangleList;
    const auto [xmin, xmax] = std::minmax_element(coverage.begin(), coverage.end(),
                                                  [](Point a, Point b) { return a.x < b.x; });
    const auto [ymin, ymax] = std::minmax_element(coverage.begin(), coverage.end(),
                                                  [](Point a, Point b) { return a.y < b.y; });
    // Bound in floating point before narrowing, including triangles that
    // extend beyond the integer raster coordinate range.
    const int min_x = int(std::clamp(std::floor(xmin->x), double(left), double(right)));
    const int min_y = int(std::clamp(std::floor(ymin->y), double(top), double(bottom)));
    const int max_x = int(std::clamp(std::ceil(xmax->x), double(left), double(right)));
    const int max_y = int(std::clamp(std::ceil(ymax->y), double(top), double(bottom)));
    std::vector<uint32_t> parameters(attributes * 12);
    for (uint32_t a = 0; a < attributes; ++a) {
      const uint32_t control = context_[0x199 + a];
      // FLAT_SHADE with OFFSET bit 5 exposes the three raw vertex values.
      const bool passthrough = (control & 0x420) == 0x420;
      if ((control & ~0x100073fu) || ((control & 0x400) && !passthrough))
        throw std::runtime_error("unsupported graphics parameter interpolation");
      for (uint32_t c = 0; c < 4; ++c) {
        std::array<float, 3> values{};
        for (uint32_t k = 0; k < 3; ++k) {
          if ((control & 32) && !passthrough) {
            values[k] = ((control >> 8) & (c == 3 ? 1u : 2u)) ? 1.0f : 0.0f;
          } else {
            const auto address = addr_calc::rdna_buffer_address(
                ring[1], ring[3], ring_stride, true, indices[k], (control & 31) * 16 + c * 4, 0, 0);
            uint32_t bits = 0;
            if (memory.read(ring_base + address.offset,
                            {reinterpret_cast<std::byte *>(&bits), sizeof(bits)}) !=
                VmAccessOutcome::Complete)
              throw std::runtime_error("graphics attribute read failed");
            values[k] = std::bit_cast<float>(bits);
          }
        }
        parameters[a * 12 + c * 3] = std::bit_cast<uint32_t>(values[0]);
        parameters[a * 12 + c * 3 + 1] =
            std::bit_cast<uint32_t>(passthrough ? values[1] : values[1] - values[0]);
        parameters[a * 12 + c * 3 + 2] =
            std::bit_cast<uint32_t>(passthrough ? values[2] : values[2] - values[0]);
      }
    }
    FragmentWave batch;
    batch.relative_layer = relative_layer;
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
          const double dx = x + 0.5 - screen[0].x, dy = y + 0.5 - screen[0].y;
          const double b1 = plane_i.at_quad(dx, dy, q);
          const double b2 = plane_j.at_quad(dx, dy, q);
          f.linear_i = b1;
          f.linear_j = b2;
          f.pull_model = {plane_iw.at_quad(dx, dy, q), plane_jw.at_quad(dx, dy, q),
                          plane_rw.at_quad(dx, dy, q)};
          const float w = 1.0f / f.pull_model[2];
          f.i = raster::multiply_perspective(f.pull_model[0], w);
          f.j = raster::multiply_perspective(f.pull_model[1], w);
          f.z = (clip_xy ? double(plane_z.at_quad(dx, dy, q))
                         : (1 - b1 - b2) * v[0].z + b1 * v[1].z + b2 * v[2].z) *
                    std::bit_cast<float>(context_[0x113]) +
                std::bit_cast<float>(context_[0x114]);
          bool inside = true;
          if (rectangle) {
            inside = px >= std::min({v[0].x, v[1].x, v[2].x}) &&
                     px < std::max({v[0].x, v[1].x, v[2].x}) &&
                     py >= std::min({v[0].y, v[1].y, v[2].y}) &&
                     py < std::max({v[0].y, v[1].y, v[2].y});
          } else {
            for (uint32_t k = 0; k < coverage.size(); ++k) {
              Point a = coverage[k], b = coverage[(k + 1) % coverage.size()];
              if (area < 0)
                std::swap(a, b);
              const double e = edge(a, b, px, py);
              const bool top_left = b.y < a.y || (b.y == a.y && b.x > a.x);
              inside &= e > 0 || (e == 0 && top_left);
            }
          }
          const bool sample_enabled = (context_[0x30e + (f.y & 1)] >> (16 * (f.x & 1))) & 1;
          f.covered =
              inside && sample_enabled && f.x >= left && f.x < right && f.y >= top && f.y < bottom;
          covered |= f.covered;
        }
        if (!covered)
          continue;
        std::copy(quad.begin(), quad.end(), batch.lanes.begin() + used);
        used += 4;
        if (used == fragment_wave_size_) {
          fragments_.push_back(std::move(batch));
          batch = FragmentWave{};
          batch.relative_layer = relative_layer;
          batch.parameters = parameters;
          used = 0;
        }
      }
    }
    if (used)
      fragments_.push_back(std::move(batch));
  }
  const bool clamp_depth = !(context_[0x19] & 1);
  const float depth_min = std::bit_cast<float>(context_[0x115]);
  const float depth_max = std::bit_cast<float>(context_[0x116]);
  if ((depth_control_ & 2) && clamp_depth &&
      (!std::isfinite(depth_min) || !std::isfinite(depth_max) || depth_min > depth_max))
    throw std::runtime_error("unsupported graphics viewport depth range");
  if (!attachments_prepared_) {
    for (const auto &color : colors_) {
      if (!color.write_mask || !color.metadata)
        continue;
      const uint32_t block_bits = 8 - std::countr_zero(color.bytes);
      const uint32_t block_width = 1u << ((block_bits + 1) / 2);
      const uint32_t block_height = 1u << (block_bits / 2);
      for (uint32_t layer = color.first_layer; layer <= color.last_layer; ++layer)
        for (uint32_t y = 0; y < color.height; y += block_height)
          for (uint32_t x = 0; x < color.width; x += block_width)
            materialize_gfx11_dcc(memory, color.base, *color.metadata, x, y, color.width,
                                  color.height, color.bytes, color.swizzle, color.pipe_aligned,
                                  layer, color.slice_size);
    }
    if ((depth_control_ & 2) && depth_metadata_) {
      uint32_t bits = depth_clear_;
      if (depth_bytes_ == 2) {
        const std::array<uint32_t, 1> component{bits};
        pack_buffer_format(7, 4, component, {reinterpret_cast<uint8_t *>(&bits), 2});
      }
      for (uint32_t y = 0; y < depth_height_; y += 8)
        for (uint32_t x = 0; x < depth_width_; x += 8)
          materialize_gfx11_htile(memory, depth_base_, *depth_metadata_, x, y, depth_width_,
                                  depth_height_, depth_bytes_, depth_swizzle_, bits);
    }
    attachments_prepared_ = true;
  }
}

void GraphicsDraw::write_outputs(const GpuVmAccess &memory) {
  const auto image_address =
      arch_ == ROCJITSU_CODE_ARCH_RDNA4 ? gfx12_image_address : gfx11_image_address;
  const bool clamp_depth = !(context_[0x19] & 1);
  const float depth_min = std::bit_cast<float>(context_[0x115]);
  const float depth_max = std::bit_cast<float>(context_[0x116]);
  for (const auto &batch : fragments_) {
    for (uint32_t lane = 0; lane < fragment_wave_size_; ++lane) {
      const auto &f = batch.lanes[lane];
      if (!f.covered)
        continue;
      if (depth_control_ & 2) {
        const auto address =
            image_address(depth_base_, f.x, f.y, depth_width_, depth_bytes_, depth_swizzle_);
        uint32_t previous_bits = 0;
        if (!address || memory.read(*address, {reinterpret_cast<std::byte *>(&previous_bits),
                                               depth_bytes_}) != VmAccessOutcome::Complete)
          throw std::runtime_error("graphics depth read failed");
        const float clamped_depth = clamp_depth ? std::clamp(f.z, depth_min, depth_max) : f.z;
        uint32_t next_bits = std::bit_cast<uint32_t>(clamped_depth);
        if (depth_bytes_ == 2) {
          const std::array<uint32_t, 1> component{next_bits};
          pack_buffer_format(7, 4, component, {reinterpret_cast<uint8_t *>(&next_bits), 2});
          next_bits &= 0xffff;
        }
        const float previous =
            depth_bytes_ == 2 ? previous_bits / 65535.0f : std::bit_cast<float>(previous_bits);
        const float depth = depth_bytes_ == 2 ? next_bits / 65535.0f : clamped_depth;
        bool pass = false;
        switch ((depth_control_ >> 4) & 7) {
        case 0:
          break;
        case 1:
          pass = depth < previous;
          break;
        case 2:
          pass = depth == previous;
          break;
        case 3:
          pass = depth <= previous;
          break;
        case 4:
          pass = depth > previous;
          break;
        case 5:
          pass = depth != previous;
          break;
        case 6:
          pass = depth >= previous;
          break;
        case 7:
          pass = true;
          break;
        }
        if (!pass)
          continue;
        if ((depth_control_ & 4) &&
            memory.write(*address, {reinterpret_cast<const std::byte *>(&next_bits),
                                    depth_bytes_}) != VmAccessOutcome::Complete)
          throw std::runtime_error("graphics depth write failed");
      }
      for (uint32_t target = 0; target < colors_.size(); ++target) {
        const auto &color = colors_[target];
        if (!color.write_mask)
          continue;
        const auto &exported = f.exports[color.export_index];
        if (!exported.mask || batch.relative_layer > color.last_layer - color.first_layer)
          continue;
        std::array<uint32_t, 4> components = exported.values;
        uint32_t component_mask = exported.mask;
        if (color.export_format == kExportFp16Abgr || color.export_format == kExportUnorm16Abgr ||
            color.export_format == kExportSnorm16Abgr || color.export_format == kExportUint16Abgr ||
            color.export_format == kExportSint16Abgr) {
          if (exported.mask & ~3u)
            throw std::runtime_error("unsupported packed graphics color export mask");
          component_mask = ((exported.mask & 1) ? 3u : 0u) | ((exported.mask & 2) ? 12u : 0u);
          for (uint32_t c = 0; c < 4; ++c) {
            const uint16_t half = exported.values[c / 2] >> (16 * (c % 2));
            if (color.export_format == kExportUint16Abgr)
              components[c] = half;
            else if (color.export_format == kExportUnorm16Abgr)
              components[c] = std::bit_cast<uint32_t>(half / 65535.0f);
            else if (color.export_format == kExportSnorm16Abgr)
              components[c] =
                  std::bit_cast<uint32_t>(std::max(static_cast<int16_t>(half) / 32767.0f, -1.0f));
            else if (color.export_format == kExportSint16Abgr)
              components[c] =
                  static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(half)));
            else
              components[c] = std::bit_cast<uint32_t>(util::f16_to_f32(half));
          }
        } else {
          const uint32_t export_mask = color.export_format == kExport32R    ? 1u
                                       : color.export_format == kExport32Gr ? 3u
                                                                            : 15u;
          if (exported.mask & ~export_mask)
            throw std::runtime_error("unsupported graphics color export mask");
        }
        const uint64_t layer_base =
            image_layer_base(arch_ == ROCJITSU_CODE_ARCH_RDNA4, color.base, color.slice_size,
                             color.first_layer + batch.relative_layer, color.bytes, color.swizzle);
        const auto address = image_address(layer_base, f.x + color.tail_x, f.y + color.tail_y,
                                           color.pitch, color.bytes, color.swizzle);
        if (!address)
          throw std::runtime_error("graphics color address is unsupported");
        std::array<uint8_t, 16> previous{}, bytes{};
        const uint32_t blend = color.blend, write_mask = color.write_mask & component_mask;
        if (!write_mask)
          continue;
        const uint32_t full_mask = (1u << color.components) - 1;
        if (((blend & kBlendEnable) || (write_mask & full_mask) != full_mask) &&
            memory.read(*address, std::as_writable_bytes(std::span{previous}.first(color.bytes))) !=
                VmAccessOutcome::Complete)
          throw std::runtime_error("graphics color read failed");
        if (blend & kBlendEnable) {
          std::array<float, 4> source{}, destination{}, constant{};
          const auto decoded = unpack_buffer_format(color.memory_format, 0xfac,
                                                    std::span{previous}.first(color.bytes));
          for (uint32_t c = 0; c < 4; ++c) {
            source[c] = std::bit_cast<float>(components[c]);
            destination[c] = std::bit_cast<float>(decoded[c]);
            constant[c] = std::bit_cast<float>(context_[0x105 + c]);
            if (color.memory_format == kBufRgba8Unorm) {
              source[c] = std::clamp(source[c], 0.0f, 1.0f);
              // UNORM destinations round to twelve significant bits before blending.
              const uint32_t normalized = decoded[c];
              destination[c] =
                  std::bit_cast<float>((normalized + 0x7ffu + ((normalized >> 12) & 1)) & ~0xfffu);
              constant[c] = std::clamp(constant[c], 0.0f, 1.0f);
              // Color blending truncates UNORM constants to twelve significant bits.
              constant[c] = std::bit_cast<float>(std::bit_cast<uint32_t>(constant[c]) & ~0xfffu);
            }
          }
          const auto widen = [](const std::array<float, 4> &values) {
            std::array<double, 4> wide{};
            std::copy(values.begin(), values.end(), wide.begin());
            return wide;
          };
          // FP16 attachments also truncate blend constants to twelve significant bits.
          if (color.memory_format == kBufRgba16Float)
            for (auto &value : constant)
              value = std::bit_cast<float>(std::bit_cast<uint32_t>(value) & ~0xfffu);
          for (uint32_t c = 0; c < 4; ++c) {
            const uint32_t control = c == 3 && (blend & kBlendSeparateAlpha) ? blend >> 16 : blend;
            float result =
                color.memory_format == kBufRgba8Unorm
                    ? blend_component(control, c, source, destination, constant)
                    : static_cast<float>(blend_component(control, c, widen(source),
                                                         widen(destination), widen(constant)));
            // Floating attachments preserve signed and out-of-range values.
            // The FP16 blend result truncates on its way back to the attachment.
            if (color.memory_format == kBufRgba16Float)
              result = util::f16_to_f32(util::f32_to_f16_rtz(result));
            components[c] = std::bit_cast<uint32_t>(result);
          }
        }
        pack_buffer_format(color.memory_format, 0xfac, components,
                           std::span{bytes}.first(color.bytes));
        if (color.srgb)
          for (uint32_t c = 0; c < 3; ++c)
            bytes[c] = srgb_color_byte(std::bit_cast<float>(components[c]));
        // Accepted color formats have equally sized components.
        const uint32_t component_bytes = color.bytes / color.components;
        for (uint32_t c = 0; c < color.components; ++c)
          if (!(write_mask & (1u << c)))
            std::copy_n(previous.begin() + c * component_bytes, component_bytes,
                        bytes.begin() + c * component_bytes);
        if (memory.write(*address, std::as_bytes(std::span{bytes}.first(color.bytes))) !=
            VmAccessOutcome::Complete)
          throw std::runtime_error("graphics color write failed");
      }
    }
  }
}

std::optional<DispatchEntry> GraphicsDraw::advance(const GpuVmAccess &memory) {
  if (fragment_stage_) {
    write_outputs(memory);
    return next_vertex_group();
  }
  finish_vertices();
  rasterize(memory);
  fragment_stage_ = true;
  if (fragments_.empty())
    return next_vertex_group();
  return fragment_dispatch();
}

} // namespace rocjitsu::amdgpu
