// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sdwa_lowering.cpp
/// @brief CDNA SDWA alternate-encoding lowering for DBT.

#include "rocjitsu/code/dbt/sdwa_lowering.h"

#include "rocjitsu/analysis/register_liveness.h"
#include "rocjitsu/code/dbt/hazard_tracker.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/instruction.h"

#include <cstring>
#include <optional>

namespace rocjitsu {
namespace {

// RDNA4 operand encoding constants.
constexpr uint16_t kInlineConst0 = 128;

// RDNA4 VOP2 opcodes.
constexpr uint8_t kOpLshlrevB32 = 24;
constexpr uint8_t kOpLshrrevB32 = 25;
constexpr uint8_t kOpAshrrevI32 = 26;
constexpr uint8_t kOpAndB32 = 27;
constexpr uint8_t kOpOrB32 = 28;

// CDNA SDWA marker and selector values.
constexpr uint32_t kCdna4SdwaSrc0Marker = 249;
constexpr uint32_t kSdwaSelByte3 = 3;
constexpr uint32_t kSdwaSelWord0 = 4;
constexpr uint32_t kSdwaSelDword = 6;

/// @brief Build a RDNA4 VOP2 instruction word.
[[nodiscard]] constexpr uint32_t build_vop2(uint8_t op, uint8_t vdst, uint16_t src0,
                                            uint8_t vsrc1) {
  return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((vdst & 0xFFu) << 17) | ((op & 0x3Fu) << 25);
}

[[nodiscard]] std::optional<uint8_t> find_free_vgpr(const RegisterLiveness &liveness,
                                                    uint64_t offset, uint8_t after = 0) {
  for (uint16_t reg = after; reg < 256; ++reg) {
    if (!liveness.is_live(offset, reg))
      return static_cast<uint8_t>(reg);
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint8_t> choose_sdwa_extract_dst(const RegisterLiveness &liveness,
                                                             uint64_t offset, uint8_t after,
                                                             std::optional<uint8_t> clobber_ok) {
  if (clobber_ok)
    return clobber_ok;
  return find_free_vgpr(liveness, offset, after);
}

[[nodiscard]] uint16_t sdwa_source_operand(uint8_t vgpr) { return 256u + vgpr; }

void emit_sdwa_extract(std::vector<uint32_t> &words, HazardTracker &hz, uint8_t dst,
                       uint8_t src_vgpr, uint32_t sel, bool sign_extend) {
  using P = HazardTracker::Pipeline;

  if (sel == kSdwaSelDword) {
    hz.emit(words, build_vop2(kOpOrB32, dst, kInlineConst0, src_vgpr), P::VALU);
    return;
  }

  if (sign_extend) {
    const uint8_t left_shift =
        (sel <= kSdwaSelByte3) ? static_cast<uint8_t>(24 - sel * 8)
                               : static_cast<uint8_t>(16 - (sel - kSdwaSelWord0) * 16);
    const uint8_t right_shift = (sel <= kSdwaSelByte3) ? 24 : 16;
    hz.emit(words, build_vop2(kOpLshlrevB32, dst, 128 + left_shift, src_vgpr), P::VALU);
    hz.emit(words, build_vop2(kOpAshrrevI32, dst, 128 + right_shift, dst), P::VALU);
    return;
  }

  const uint32_t right_shift = (sel <= kSdwaSelByte3) ? sel * 8 : (sel - kSdwaSelWord0) * 16;
  if (right_shift != 0)
    hz.emit(words, build_vop2(kOpLshrrevB32, dst, 128 + right_shift, src_vgpr), P::VALU);
  else if (dst != src_vgpr)
    hz.emit(words, build_vop2(kOpOrB32, dst, kInlineConst0, src_vgpr), P::VALU);

  const uint32_t mask = (sel <= kSdwaSelByte3) ? 0xFFu : 0xFFFFu;
  hz.emit2(words, build_vop2(kOpAndB32, dst, 255, dst), mask, P::VALU);
}

} // namespace

bool is_cdna4_vop2_sdwa_form(const Instruction &inst) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() != sizeof(cdna4::Vop2VopSdwaMachineInst))
    return false;

  cdna4::Vop2MachineInst vop2{};
  std::memcpy(&vop2, raw, sizeof(vop2));
  return vop2.src0 == kCdna4SdwaSrc0Marker;
}

std::vector<uint32_t> lower_cdna4_vop2_sdwa_to_rdna4(const Instruction &inst, uint64_t offset,
                                                     const RegisterLiveness &liveness,
                                                     uint16_t dst_opcode, uint32_t ext_word) {
  if (!is_cdna4_vop2_sdwa_form(inst))
    return {};

  uint32_t raw_words[2] = {inst.raw_encoding()[0], ext_word};
  cdna4::Vop2VopSdwaMachineInst sdwa{};
  std::memcpy(&sdwa, raw_words, sizeof(sdwa));

  // This lowering handles SDWA as source extraction feeding the ordinary RDNA4
  // VOP2 operation. Destination sub-dword merge, modifiers, clamp, and scalar
  // SDWA sources need distinct instruction sequences, so reject them here
  // instead of preserving the second SDWA word as executable code.
  if (sdwa.dst_sel != kSdwaSelDword || sdwa.clamp || sdwa.omod || sdwa.src0_neg ||
      sdwa.src0_abs || sdwa.src1_neg || sdwa.src1_abs || sdwa.s0 || sdwa.s1)
    return {};
  if (sdwa.src0_sel > kSdwaSelDword || sdwa.src1_sel > kSdwaSelDword)
    return {};

  std::vector<uint32_t> words;
  HazardTracker hz;

  uint16_t src0 = sdwa_source_operand(sdwa.vsrc0);
  uint8_t src1 = sdwa.vsrc1;
  uint8_t tmp_floor = 0;

  if (sdwa.src0_sel != kSdwaSelDword) {
    const bool can_clobber_vdst = sdwa.vsrc0 == sdwa.vdst && sdwa.vsrc1 != sdwa.vdst;
    auto tmp = choose_sdwa_extract_dst(liveness, offset, tmp_floor,
                                       can_clobber_vdst
                                           ? std::optional<uint8_t>(static_cast<uint8_t>(sdwa.vdst))
                                           : std::nullopt);
    if (!tmp)
      return {};
    if (*tmp >= tmp_floor)
      tmp_floor = *tmp + 1;
    emit_sdwa_extract(words, hz, *tmp, sdwa.vsrc0, sdwa.src0_sel, sdwa.src0_sext);
    src0 = sdwa_source_operand(*tmp);
  }

  if (sdwa.src1_sel != kSdwaSelDword) {
    const bool can_clobber_vdst = sdwa.vsrc1 == sdwa.vdst && sdwa.vsrc0 != sdwa.vdst;
    auto tmp = choose_sdwa_extract_dst(liveness, offset, tmp_floor,
                                       can_clobber_vdst
                                           ? std::optional<uint8_t>(static_cast<uint8_t>(sdwa.vdst))
                                           : std::nullopt);
    if (!tmp)
      return {};
    emit_sdwa_extract(words, hz, *tmp, sdwa.vsrc1, sdwa.src1_sel, sdwa.src1_sext);
    src1 = *tmp;
  }

  hz.emit(words, build_vop2(static_cast<uint8_t>(dst_opcode), sdwa.vdst, src0, src1),
          HazardTracker::Pipeline::VALU);
  return words;
}

} // namespace rocjitsu
