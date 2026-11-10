// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_resource.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/vm/amdgpu/image_addressing.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "util/result.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <optional>

namespace rocjitsu::amdgpu {

/// An ordinary T# can reference sparse storage with MIN_LOD_WARN set to zero.
inline bool image_may_have_sparse_pages(const Wavefront &wave) {
  if (!wave.address_space())
    return false;
  const auto vm = wave.snapshot_vm_access();
  if (!vm || !vm->info().ready)
    return true;
  const auto ranges = vm->nonresident_prt_ranges();
  return ranges && !ranges->empty();
}

/// @brief Encoded image access controls and coordinate VGPR selectors.
struct ImageFields {
  uint32_t dmask = 0;
  uint32_t dim = 0;
  bool a16 = false;
  bool d16 = false;
  bool r128 = false;
  bool tfe = false;
  bool lwe = false;
  bool da = false;
  bool unorm = false;
  bool mip = false;    ///< The address tuple includes a view-relative mip level.
  bool gather = false; ///< Return four selected texel components.
  bool gradient16 = false;
  bool offset = false;
  bool lod_bias = false;
  bool lod_clamp = false;
  bool compare = false;
  bool horizontal = false;
  uint8_t packed_texels = 0;
  uint32_t pixels = 0;    // Packed/raw and horizontal multi-texel load/store family.
  bool msaa_load = false; // GFX10.3 multi-fragment single-channel load.
  bool raw = false;
  bool sign_extend = false;
};

/// Resolve the consumed coordinate words, including the individual VGPR0 substitutions on GFX11.5.
inline bool resolve_image_coordinates(const Wavefront &wf, ImageAddressRegisters &addresses,
                                      uint32_t count) {
  if (count > addresses.size())
    return false;
  for (uint32_t i = 0; i < count; ++i) {
    if (addresses[i] < wf.num_vgprs())
      continue;
    if (!has_gfx11_plus_image_operands(wf.cu().arch()))
      return false;
    addresses[i] = gfx115_source_vgpr(addresses[i], wf.num_vgprs());
  }
  return true;
}

/// @brief Prepare and base-check each mip once per image instruction.
class ImageMipAddressCache {
public:
  ImageMipAddressCache(ImageAddressConfig config, ImageLayout layout, uint64_t base,
                       VectorMemState::BlockCompressedImage *bc = nullptr)
      : config_(config), layout_(layout), base_(base), bc_(bc) {
    if (bc_) {
      layout_.width = (layout_.width + 3) / 4;
      layout_.height = (layout_.height + 3) / 4;
      layout_.pitch = (layout_.pitch + 3) / 4;
    }
  }

  bool contains(const ImageAddressing &addressing, unsigned x, unsigned y, unsigned z) const {
    return addressing.contains(bc_ ? x / 4 : x, bc_ ? y / 4 : y, z);
  }

  uint64_t offset(const ImageAddressing &addressing, unsigned x, unsigned y, unsigned z,
                  unsigned lane, unsigned tap = 0, unsigned sample = 0) {
    if (!bc_)
      return addressing.byte_offset(x, y, z, sample);
    if (bc_->texels.size() <= tap)
      bc_->texels.resize(tap + 1);
    bc_->texels[tap][lane] = (x % 4) + (y % 4) * 4;
    return addressing.byte_offset(x / 4, y / 4, z);
  }

  const ImageAddressing *get(uint32_t level) {
    if (level >= levels_.size())
      return nullptr;
    std::optional<ImageAddressing> &slot = levels_[level];
    if (!slot) {
      slot = ImageAddressing::prepare(config_, layout_, level);
      if (!slot || !slot->has_supported_base(base_)) {
        slot.reset();
        return nullptr;
      }
    }
    return &*slot;
  }

private:
  ImageAddressConfig config_;
  ImageLayout layout_;
  uint64_t base_;
  VectorMemState::BlockCompressedImage *bc_;
  std::array<std::optional<ImageAddressing>, kMaxImageMipLevels> levels_;
};

/// W7900 and R9700 repeat stored image channels for XYZW selectors;
/// literal zero and one selectors are independent of storage channel count.
inline uint32_t image_load_selectors(uint32_t selectors, const BufferFormat &format) {
  const uint32_t count = format.component_count();
  if (!count)
    return selectors;
  for (uint32_t component = 0; component < 4; ++component) {
    const uint32_t shift = 3 * component;
    const uint32_t select = (selectors >> shift) & 7;
    if (select >= 4)
      selectors = (selectors & ~(7u << shift)) | ((4 + (select - 4) % count) << shift);
  }
  return selectors;
}

/// @brief Decode and qualify the descriptor and return its formatted-transfer state.
/// @returns A zero-width resource for an unbacked descriptor, null type, or INVALID
/// format. The instruction completes with an empty memory request in those cases.
inline util::FailureOr<ImageResource>
prepare_image_resource(Wavefront &wf, VectorMemState &state, uint32_t resource_reg,
                       ImageFields fields, bool result_in_range = true, bool sampled = false) {
  const auto fail = [&](InstructionExecutionError error) {
    wf.report_instruction_execution_error(error);
    return util::Result::failure();
  };
  state.wf_size = wf.wf_size();
  state.wg_id = wf.wg_id();
  state.wf_id = wf.wf_id();
  state.exec_mask = result_in_range ? wf.exec() : 0;
  state.lane_mask = 0;
  state.per_lane_addr.fill(0);
  state.num_elems = 1;
  state.elem_size = 1;
  state.buffer_d16 = fields.d16;
  state.buffer_components = static_cast<uint32_t>(std::popcount(fields.dmask & 15u));
  const rj_code_arch_t arch = wf.cu().arch();
  if (!arch_has_image_memory_instructions(arch))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const bool gfx9 = arch_is_cdna_4_or_lower(arch);
  state.buffer_format_encoding = gfx9                               ? BufferFormatEncoding::Gfx9
                                 : arch == ROCJITSU_CODE_ARCH_RDNA1 ? BufferFormatEncoding::Rdna1
                                 : arch == ROCJITSU_CODE_ARCH_RDNA2 ? BufferFormatEncoding::Rdna2
                                                                    : BufferFormatEncoding::Gfx11;
  const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  if (!gfx10 && (fields.r128 || fields.tfe || fields.lwe))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (state.buffer_components == 0 ||
      ((!state.is_load || state.atomic_op != AtomicOp::NONE) && !fields.unorm) ||
      (fields.raw && fields.d16))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  state.check_texel_residency = gfx10 || (has_gfx11_plus_image_operands(arch) && !state.is_load &&
                                          state.atomic_op == AtomicOp::NONE);
  if (state.is_load && (gfx10 || (sampled && arch == ROCJITSU_CODE_ARCH_RDNA4))) {
    const bool status = fields.tfe || (sampled && fields.lwe);
    uint32_t count = fields.gather && !fields.packed_texels ? 4u : state.buffer_components;
    if (fields.d16)
      count = (count + 1) / 2;
    if (atomic_uses_two_sources(state.atomic_op))
      count /= 2;
    // VDATA and the optional trailing status are one architectural result.
    if (!wf.vgpr_alloc().contains(state.dst_reg_base, count + status)) {
      state.exec_mask = 0;
      // GFX10.3 still consumes the scalar descriptor of an issued memory instruction.
      if (arch != ROCJITSU_CODE_ARCH_RDNA2)
        return ImageResource{};
    }
    if (status && state.exec_mask) {
      state.texel_status = std::make_unique<TexelStatus>();
      state.texel_status->report_failure = fields.tfe;
    }
  }

  const uint32_t descriptor_words = fields.r128 ? 4 : 8;
  if (!scalar_selector_range_is_backed(wf, resource_reg, descriptor_words))
    return ImageResource{};

  std::array<uint32_t, 8> words{};
  for (uint32_t i = 0; i < descriptor_words; ++i)
    words[i] = read_scalar_selector(wf, resource_reg + i);
  const ImageResource resource = decode_image_resource(words, arch, fields.r128);
  if (!gfx9 && !gfx10 && state.is_load && state.exec_mask && !state.image_status &&
      (resource.min_lod_warn || image_may_have_sparse_pages(wf))) {
    state.image_status = std::make_unique<ImageFeedback>();
    state.image_status->warning_threshold = resource.min_lod_warn;
  }
  const bool msaa = resource.type == kImage2dMsaa || resource.type == kImage2dMsaaArray;

  if (resource.type < kImage1d || resource.data_format == 0)
    return ImageResource{};

  if (fields.r128 && resource.type != kImage1d && resource.type != kImage2d &&
      resource.type != kImage2dMsaa)
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  if (!gfx10 && (msaa || (!gfx9 && fields.dim >= kImageDim2dMsaa)))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if ((msaa && (sampled || fields.mip || fields.pixels || resource.base_level ||
                resource.last_level > 3 || fields.dim + kImage1d != resource.type)) ||
      (!msaa && (fields.msaa_load || fields.dim >= kImageDim2dMsaa) && gfx10))
    return fail(InstructionExecutionError::UnsupportedOperandValue);

  if (resource.compressed || resource.base_level > resource.last_level ||
      (!msaa && resource.last_level > resource.max_level) ||
      resource.max_level >= kMaxImageMipLevels)
    return fail(InstructionExecutionError::UnimplementedInstruction);

  // Descriptor-only RESINFO supports UAV3D views; memory accesses to
  // those views and scaled GFX11/12 resources remain unqualified.
  if (resource.uav3d || resource.depth_scale || resource.height_scale || resource.width_scale)
    return fail(InstructionExecutionError::UnimplementedInstruction);

  // DMASK packs supplied shader components into consecutive registers (or
  // halves for D16). Loads compact their selectors to produce that tuple.
  // RDNA stores retain the descriptor's shader-component slots and scatter
  // their packed inputs back into those slots before applying the full
  // mapping. CDNA2 retains its compacted-selector store behavior.
  const uint32_t descriptor_selectors = resource.selectors;
  if (state.is_load || gfx9) {
    uint32_t selectors = 0;
    for (uint32_t component = 0, packed = 0; component < 4; ++component)
      if (fields.dmask & (1u << component))
        selectors |= ((descriptor_selectors >> (3 * component)) & 7) << (3 * packed++);
    state.buffer_selectors = selectors;
  } else {
    state.buffer_selectors = descriptor_selectors;
    const uint32_t mask = fields.dmask & 15u;
    state.image_store_component_mask =
        arch == ROCJITSU_CODE_ARCH_RDNA1 && fields.d16 ? (1u << std::popcount(mask)) - 1 : mask;
  }

  // The six-bit image DFMT must fit the buffer decoder's four-bit packed key.
  if (gfx9 && resource.data_format > 15)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  state.buffer_format = resource.data_format | (resource.num_format << 4);
  state.image_srgb = false;
  constexpr uint32_t kGfx9SrgbNumberFormat = 9;
  if (gfx9 && fields.raw) {
    // PCK and integer image atomics use the descriptor's texel width but
    // bypass numeric conversion, including sRGB and signed interpretation.
    state.buffer_format = resource.data_format | (4u << 4);
  } else if (gfx9 && resource.num_format == kGfx9SrgbNumberFormat) {
    state.image_srgb = true;
    state.buffer_format = resource.data_format; // Decode channels as UNORM first.
  } else if (!gfx9) {
    const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
    state.buffer_format = image_srgb_storage_format(resource.data_format, gfx10);
    state.image_srgb = state.buffer_format != resource.data_format;
  }
  if (state.image_srgb && !state.is_load && !fields.raw)
    return fail(InstructionExecutionError::UnimplementedInstruction);
  const util::FailureOr<bool> format =
      gfx10 ? prepare_gfx10_image_format(state) : prepare_buffer_format_state(state);
  if (state.image_bc &&
      (!state.is_load || fields.raw || msaa || state.atomic_op != AtomicOp::NONE ||
       (resource.max_level != 0 &&
        (!std::has_single_bit(resource.width) || !std::has_single_bit(resource.height)))))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  // Buffer decoder failure does not imply an illegal image format: images
  // also support formats such as SRGB that the buffer table does not model.
  if (format.failed())
    return fail(InstructionExecutionError::UnimplementedInstruction);
  if (!format.value())
    return ImageResource{};
  if (state.is_load && !gfx9 && !gfx10 && !fields.raw)
    state.buffer_selectors =
        image_load_selectors(state.buffer_selectors, state.decoded_buffer_format);

  const bool layered_view =
      gfx10 && (resource.type == kImage1dArray || resource.type == kImage2dArray ||
                resource.type == kImage2dMsaaArray || resource.type == kImageCube);
  if (resource.linked_resource || (!gfx10 && resource.type == kImageCube) ||
      (resource.type > kImage2dArray && !msaa) || resource.array_pitch ||
      (resource.base_array && !layered_view) ||
      (layered_view && resource.base_array >= resource.depth))
    return fail(InstructionExecutionError::UnimplementedInstruction);

  if (!gfx10 && !gfx9 && fields.raw) {
    state.image_srgb = false;
    auto &raw = state.decoded_buffer_format;
    raw.number = fields.sign_extend ? BufferNumberFormat::Sint : BufferNumberFormat::Uint;
    state.buffer_selectors = 0;
    for (unsigned channel = 0, packed = 0; channel < 4; ++channel) {
      raw.widths[channel] =
          state.elem_size > channel * 4 ? std::min(4u, state.elem_size - channel * 4) * 8 : 0;
      if (fields.dmask & (1u << channel))
        state.buffer_selectors |= (4u + channel) << (3 * packed++);
    }
    state.image_store_component_mask = fields.dmask;
  }
  // CDNA2 preserves its legacy compacted-selector acceptance; only RDNA's
  // full-selector/scatter path requires a qualified inverse mapping.
  if (!gfx9 && !state.is_load && !fields.raw && state.atomic_op == AtomicOp::NONE &&
      !image_store_selectors_supported(state.decoded_buffer_format, descriptor_selectors, gfx10))
    return fail(InstructionExecutionError::UnimplementedInstruction);

  return resource;
}

/// @brief Prepare IMAGE_LOAD/STORE with optional view-relative mip operands for the vector memory
/// pipeline.
/// @details The descriptor selects format, bounds, row pitch and tiling;
/// coordinates may come from a consecutive tuple or NSA VGPRs.
inline util::Result prepare_image_access(Wavefront &wf, VectorMemState &state,
                                         uint32_t resource_reg, ImageAddressRegisters vaddr,
                                         ImageFields fields, bool result_in_range = true) {
  const auto fail = [&](InstructionExecutionError error) {
    wf.report_instruction_execution_error(error);
    return util::Result::failure();
  };
  const util::FailureOr<ImageResource> prepared =
      prepare_image_resource(wf, state, resource_reg, fields, result_in_range);
  if (prepared.failed())
    return util::Result::failure();
  const ImageResource &resource = prepared.value();
  if (fields.msaa_load && resource.width) {
    if (state.decoded_buffer_format.widths[1])
      return fail(InstructionExecutionError::UnsupportedOperandValue);
    state.image_sample = std::make_unique<ImageSampleAccess>();
    state.image_sample->taps.resize(4);
    state.image_sample->transfer.emplace(fields.dmask, 1, false, false);
    state.buffer_selectors = resource.selectors;
  }
  if (fields.pixels && resource.width) {
    const uint32_t slots = fields.raw ? (state.elem_size + 3) / 4
                                      : std::count_if(state.decoded_buffer_format.widths.begin(),
                                                      state.decoded_buffer_format.widths.end(),
                                                      [](auto width) { return width != 0; });
    if (slots == 0 || slots * fields.pixels > 4 || (fields.raw && fields.d16))
      return fail(InstructionExecutionError::UnsupportedOperandValue);
    // Packed data uses raw DWORD requests so DMASK can suppress individual
    // words of a multi-DWORD texel on stores without touching their bytes.
    state.image_sample = std::make_unique<ImageSampleAccess>();
    state.image_sample->taps.resize(fields.pixels * (fields.raw ? slots : 1));
    state.image_sample->transfer.emplace(fields.dmask, fields.raw ? 1u : slots, fields.raw,
                                         fields.sign_extend);
    state.buffer_selectors = resource.selectors;
  }
  if (!resource.width)
    return util::Result::success();
  const uint64_t exec = state.exec_mask;
  const rj_code_arch_t arch = wf.cu().arch();
  const bool gfx9 = arch_is_cdna_4_or_lower(arch);
  const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  const ImageCoordinateShape shape = image_coordinate_shape(resource.type, fields.da || !gfx9);
  const bool msaa = resource.type >= kImage2dMsaa;
  const bool layered = shape.layered || resource.type == kImage2dMsaaArray;
  // Execution consumes the spatial prefix selected by the resource type,
  // which can be shorter than the DIM-encoded coordinate list. The compatibility
  // checks below require DIM to supply enough coordinates, not an equal count;
  // for example, a 2D resource with 2D_ARRAY DIM uses X/Y and ignores layer.
  const bool has_y = shape.has_y;
  const bool has_slice = shape.has_slice || resource.type == kImage2dMsaaArray;
  const uint32_t slice_coordinate =
      image_transfer_slice_coordinate(resource.type, fields.dim, gfx9, has_y);
  const uint32_t coordinates =
      image_transfer_coordinate_count(resource.type, has_y, has_slice, slice_coordinate);
  if (!gfx9) {
    const uint32_t dim = fields.dim & 7u;
    if (!image_dim_has_resource_layer(dim, resource.type))
      return fail(InstructionExecutionError::UnimplementedInstruction);
    uint32_t dim_spatial_coordinates = 0;
    switch (dim) {
    case kImageDim1d:
    case kImageDim1dArray:
      dim_spatial_coordinates = 1;
      break;
    case kImageDimCube:
    case kImageDim2dMsaa:
    case kImageDim2dMsaaArray:
    case kImageDim2d:
    case kImageDim2dArray:
      dim_spatial_coordinates = 2;
      break;
    case kImageDim3d:
      dim_spatial_coordinates = 3;
      break;
    default:
      return fail(InstructionExecutionError::UnimplementedInstruction);
    }
    // CLR accesses thin 3D copies with a 2D_ARRAY instruction. Both it and
    // 3D DIM supply X/Y plus the slice coordinate consumed by that resource.
    const uint32_t resource_spatial_coordinates = has_y ? 2 : 1;
    const uint32_t dim_coordinates = image_coordinate_count(dim);
    if (dim_spatial_coordinates < resource_spatial_coordinates || dim_coordinates < coordinates)
      return fail(InstructionExecutionError::UnimplementedInstruction);
  }
  const uint32_t mip_coordinate = gfx9 ? coordinates : image_coordinate_count(fields.dim);
  const uint32_t inputs = fields.mip ? mip_coordinate + 1 : coordinates;
  const uint32_t registers = fields.a16 ? (inputs + 1) / 2 : inputs;
  if (!resolve_image_coordinates(wf, vaddr, registers))
    return fail(InstructionExecutionError::UnsupportedOperandValue);
  RegisterAccess regs(wf);
  const uint32_t depth = layered || resource.type == kImage3d ? resource.depth : 1;
  const ImageLayout layout{.width = resource.width,
                           .height = has_y ? resource.height : 1u,
                           .depth = depth,
                           .pitch = resource.layout_pitch(arch, /*sample_2d=*/false),
                           .element_bytes = state.elem_size,
                           .swizzle_mode = resource.swizzle_mode,
                           .dimensions = resource.type == kImage3d ? 3u
                                         : has_y                   ? 2u
                                                                   : 1u,
                           .mip_levels = msaa ? 1u : resource.max_level + 1,
                           .samples = msaa ? 1u << resource.last_level : 1u};
  ImageMipAddressCache addresses(wf.cu().image_address_config(), layout, resource.base,
                                 state.image_bc.get());
  if (!addresses.get(resource.base_level))
    return fail(InstructionExecutionError::UnimplementedInstruction);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (uint64_t{1} << lane)))
      continue;
    std::array<uint32_t, 4> coords{};
    for (uint32_t i = 0; i < inputs; ++i) {
      const uint32_t reg = vaddr[fields.a16 ? i / 2 : i];
      const uint32_t bits =
          regs.read_vgpr(wf.vgpr_alloc().base + reg, lane,
                         image_coordinate_component_bytes(i, !gfx9 && fields.a16));
      coords[i] = fields.a16 ? static_cast<uint16_t>(bits >> (16 * (i % 2))) : bits;
    }
    const uint32_t x = coords[0];
    const uint32_t y = has_y ? coords[1] : 0;
    const uint32_t relative_slice = has_slice ? coords[slice_coordinate] : 0;
    if (layered && relative_slice >= depth - resource.base_array)
      continue;
    const uint32_t slice = relative_slice + (layered ? resource.base_array : 0);
    const uint64_t level64 =
        uint64_t{resource.base_level} + (fields.mip ? coords[mip_coordinate] : 0);
    if (level64 > resource.last_level)
      continue;
    const uint32_t level = level64;
    if (state.texel_status)
      state.texel_status->lod[lane] = level;
    if (x >= std::max(1u, resource.width >> level) || y >= std::max(1u, resource.height >> level) ||
        slice >= (resource.type == kImage3d ? std::max(1u, depth >> level) : depth))
      continue;
    const ImageAddressing *addressing = addresses.get(level);
    if (!addressing)
      return fail(InstructionExecutionError::UnimplementedInstruction);
    if (!addresses.contains(*addressing, x, y, slice))
      continue;
    const uint32_t sample = msaa ? coords[coordinates - 1] : 0;
    if (sample >= layout.samples)
      continue;
    if (fields.msaa_load) {
      // GFX10.3 DMASK selects the fragments to return, unlike GFX11's
      // gather-style channel selector. VDATA compacts the requested fragments.
      for (uint32_t i = 0; i < 4 && sample + i < layout.samples; ++i) {
        if (!(fields.dmask & (1u << i)))
          continue;
        auto &tap = state.image_sample->taps[i];
        tap.addresses[lane] = addr_calc::buffer_virtual_address(
            resource.base + addressing->byte_offset(x, y, slice, sample + i));
        tap.lane_mask |= uint64_t{1} << lane;
        state.lane_mask |= uint64_t{1} << lane;
      }
    } else if (fields.pixels) {
      const auto &transfer = *state.image_sample->transfer;
      const uint32_t local_mask = (1u << transfer.slots_per_texel) - 1;
      const uint32_t taps_per_pixel = state.image_sample->taps.size() / fields.pixels;
      const uint32_t width = std::max(1u, resource.width >> level);
      for (unsigned pixel = 0; pixel < fields.pixels && pixel < width - x; ++pixel) {
        if (!addresses.contains(*addressing, x + pixel, y, slice))
          continue;
        const uint64_t base =
            resource.base + addresses.offset(*addressing, x + pixel, y, slice, lane, pixel);
        for (unsigned word = 0; word < taps_per_pixel; ++word) {
          const unsigned index = pixel * taps_per_pixel + word;
          if (!(fields.dmask & (local_mask << (index * transfer.slots_per_texel))))
            continue;
          auto &tap = state.image_sample->taps[index];
          tap.addresses[lane] = addr_calc::buffer_virtual_address(base + word * 4u);
          tap.lane_mask |= uint64_t{1} << lane;
          state.lane_mask |= uint64_t{1} << lane;
        }
      }
    } else {
      const uint64_t offset = addresses.offset(*addressing, x, y, slice, lane, 0, sample);
      state.per_lane_addr[lane] = addr_calc::buffer_virtual_address(resource.base + offset);
      state.lane_mask |= uint64_t{1} << lane;
    }
  }
  if (gfx10 && fields.raw)
    state.elem_size = std::min(4u, state.elem_size);
  return util::Result::success();
}

} // namespace rocjitsu::amdgpu
