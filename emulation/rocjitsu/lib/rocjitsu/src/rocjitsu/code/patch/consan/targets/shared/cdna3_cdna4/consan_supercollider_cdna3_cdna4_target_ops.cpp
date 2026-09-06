// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_cdna3_cdna4_target_ops.cpp
/// @brief CDNA3/CDNA4 recipes consumed by SuperCollider.

#include "rocjitsu/code/patch/consan/targets/consan_supercollider_target_ops_internal.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"

#include <cstring>

namespace rocjitsu::consan_sc_target_detail {

std::optional<uint32_t> build_cdna3_cdna4_ds_load_word0(const ConSanAccessLoweringForm &form,
                                                        uint32_t original_word0) {
  uint32_t base = 0;
  if (form.kind == ConSanAccessLoweringFormKind::NativeTwoRange) {
    switch (original_word0 & 0xFFFF0000u) {
    case 0xD81C0000u:
      base = 0xD86E0000u;
      break;
    case 0xD81E0000u:
      base = 0xD8700000u;
      break;
    case 0xD89C0000u:
      base = 0xD8EE0000u;
      break;
    case 0xD89E0000u:
      base = 0xD8F00000u;
      break;
    default:
      return std::nullopt;
    }
  } else {
    switch (form.element_width_bits) {
    case 8:
      base = cdna4::build_ds(cdna4::kDsReadU8Ds)[0];
      break;
    case 16:
      base = 0xD8780000u;
      break;
    case 32:
      base = 0xD86C0000u;
      break;
    case 64:
      base = 0xD8EC0000u;
      break;
    case 96:
      base = 0xD9FC0000u;
      break;
    case 128:
      base = 0xD9FE0000u;
      break;
    default:
      return std::nullopt;
    }
  }
  constexpr uint32_t kDsOffsetMask = 0x0000FFFFu;
  return base | (original_word0 & kDsOffsetMask);
}

std::optional<ConSanScDirectToLdsTransfer> build_cdna3_cdna4_direct_to_lds_transfer(
    std::array<uint32_t, 2> original_words, uint32_t width_bits, uint16_t address_vgpr,
    uint16_t payload_vgpr, uint16_t readback_vgpr, rj_code_arch_t arch) {
  const uint16_t dwords = static_cast<uint16_t>(width_bits / 32u);
  if ((width_bits != 32u && width_bits != 96u && width_bits != 128u) || address_vgpr > 255u ||
      payload_vgpr > 256u - dwords || readback_vgpr > 256u - dwords)
    return std::nullopt;

  cdna4::MubufMachineInst load{};
  static_assert(sizeof(load) == sizeof(original_words));
  std::memcpy(&load, original_words.data(), sizeof(load));
  const uint16_t expected_load_op = width_bits == 32u   ? cdna4::kBufferLoadDwordMubuf
                                    : width_bits == 96u ? cdna4::kBufferLoadDwordx3Mubuf
                                                        : cdna4::kBufferLoadDwordx4Mubuf;
  if (load.lds == 0u || load.vdata != 0u || load.op != expected_load_op)
    return std::nullopt;
  load.lds = 0u;
  load.vdata = payload_vgpr;
  std::array<uint32_t, 2> global_load{};
  std::memcpy(global_load.data(), &load, sizeof(load));

  if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
    const uint16_t write_op = width_bits == 32u   ? cdna3::kDsWriteB32Ds
                              : width_bits == 96u ? cdna3::kDsWriteB96Ds
                                                  : cdna3::kDsWriteB128Ds;
    const uint16_t read_op = width_bits == 32u   ? cdna3::kDsReadB32Ds
                             : width_bits == 96u ? cdna3::kDsReadB96Ds
                                                 : cdna3::kDsReadB128Ds;
    return ConSanScDirectToLdsTransfer{
        .global_load = global_load,
        .lds_write = cdna3::build_ds(write_op, {.addr = static_cast<uint8_t>(address_vgpr),
                                                .data0 = static_cast<uint8_t>(payload_vgpr)}),
        .lds_readback = cdna3::build_ds(read_op, {.addr = static_cast<uint8_t>(address_vgpr),
                                                  .vdst = static_cast<uint8_t>(readback_vgpr)}),
    };
  }
  if (arch != ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  const uint16_t write_op = width_bits == 32u   ? cdna4::kDsWriteB32Ds
                            : width_bits == 96u ? cdna4::kDsWriteB96Ds
                                                : cdna4::kDsWriteB128Ds;
  const uint16_t read_op = width_bits == 32u   ? cdna4::kDsReadB32Ds
                           : width_bits == 96u ? cdna4::kDsReadB96Ds
                                               : cdna4::kDsReadB128Ds;
  return ConSanScDirectToLdsTransfer{
      .global_load = global_load,
      .lds_write = cdna4::build_ds(write_op, {.addr = static_cast<uint8_t>(address_vgpr),
                                              .data0 = static_cast<uint8_t>(payload_vgpr)}),
      .lds_readback = cdna4::build_ds(read_op, {.addr = static_cast<uint8_t>(address_vgpr),
                                                .vdst = static_cast<uint8_t>(readback_vgpr)}),
  };
}

std::optional<std::array<uint32_t, 2>>
build_cdna3_cdna4_accvgpr_read_b32(uint16_t dst_vgpr, uint16_t src_accvgpr, rj_code_arch_t arch) {
  if (dst_vgpr > 255u || src_accvgpr > 255u)
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
    return cdna3::build_vop3p(cdna3::kVAccvgprReadVop3p,
                              {.vdst = static_cast<uint8_t>(dst_vgpr),
                               .op_sel_hi_2 = 1u,
                               .src0 = static_cast<uint16_t>(256u + src_accvgpr),
                               .op_sel_hi = 3u});
  }
  if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
    return cdna4::build_vop3p(cdna4::kVAccvgprReadVop3p,
                              {.vdst = static_cast<uint8_t>(dst_vgpr),
                               .op_sel_hi_2 = 1u,
                               .src0 = static_cast<uint16_t>(256u + src_accvgpr),
                               .op_sel_hi = 3u});
  }
  return std::nullopt;
}

std::optional<std::array<uint32_t, 3>>
build_cdna3_cdna4_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits,
                                       uint16_t vdst) {
  uint32_t load_op = 0;
  switch (width_bits) {
  case 8:
    load_op = cdna4::kFlatLoadUbyteFlat;
    break;
  case 16:
    load_op = cdna4::kFlatLoadUshortFlat;
    break;
  case 32:
    load_op = cdna4::kFlatLoadDwordFlat;
    break;
  case 64:
    load_op = cdna4::kFlatLoadDwordx2Flat;
    break;
  case 128:
    load_op = cdna4::kFlatLoadDwordx4Flat;
    break;
  default:
    return std::nullopt;
  }
  constexpr uint32_t kFlatOpMask = 0x7fu << 18u;
  constexpr uint32_t kFlatDataMask = 0xffu << 8u;
  words[0] = (words[0] & ~kFlatOpMask) | (load_op << 18u);
  words[1] = (words[1] & ~kFlatDataMask & 0x00FFFFFFu) | (static_cast<uint32_t>(vdst) << 24u);
  return words;
}

} // namespace rocjitsu::consan_sc_target_detail
