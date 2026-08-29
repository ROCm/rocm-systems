// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_rdna4_target_ops.cpp
/// @brief RDNA4 recipes consumed by SuperCollider.

#include "rocjitsu/code/patch/consan/consan_supercollider_target_ops_internal.h"

#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"

#include <cstring>

namespace rocjitsu::consan_sc_target_detail {
namespace {

[[nodiscard]] std::optional<uint8_t> flat_load_op_for_width(uint32_t width_bits) {
  switch (width_bits) {
  case 8:
    return rdna4::kFlatLoadU8Vflat;
  case 16:
    return rdna4::kFlatLoadU16Vflat;
  case 32:
    return rdna4::kFlatLoadB32Vflat;
  case 64:
    return rdna4::kFlatLoadB64Vflat;
  case 128:
    return rdna4::kFlatLoadB128Vflat;
  default:
    return std::nullopt;
  }
}

} // namespace

std::optional<std::array<uint32_t, 3>> retarget_rdna4_flat_load_vdst(std::array<uint32_t, 3> words,
                                                                     uint16_t vdst) {
  rdna4::VflatMachineInst inst{};
  std::memcpy(&inst, words.data(), sizeof(inst));
  inst.vdst = vdst;
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

std::optional<std::array<uint32_t, 3>>
build_rdna4_flat_load_from_store(std::array<uint32_t, 3> words, uint32_t width_bits,
                                 uint16_t vdst) {
  const auto load_op = flat_load_op_for_width(width_bits);
  if (!load_op)
    return std::nullopt;
  rdna4::VflatMachineInst inst{};
  std::memcpy(&inst, words.data(), sizeof(inst));
  inst.op = *load_op;
  inst.vdst = vdst;
  inst.vsrc = 0;
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

} // namespace rocjitsu::consan_sc_target_detail
