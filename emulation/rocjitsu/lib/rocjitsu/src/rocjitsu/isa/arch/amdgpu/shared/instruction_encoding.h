// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instruction_encoding.h
/// @brief Lightweight AMDGPU instruction-encoding helpers.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_INSTRUCTION_ENCODING_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_INSTRUCTION_ENCODING_H_

#include <cstdint>
#include <string>

namespace rocjitsu {
namespace amdgpu {

/// @brief VOP1/VOP2 src0 encoding values that indicate DPP or SDWA modifiers.
constexpr uint32_t SRC_SDWA = 249;
constexpr uint32_t SRC_DPP = 250;
constexpr uint32_t SRC_DPP8_FI_0 = 233;
constexpr uint32_t SRC_DPP8_FI_1 = 234;
constexpr uint32_t SRC_DPP8_LO = SRC_DPP8_FI_0;
constexpr uint32_t SRC_DPP8_HI = SRC_DPP8_FI_1;

namespace dpp {

/// @brief DPP control value ranges encoded in VOP instruction modifiers.
enum DppCtrl : uint32_t {
  QUAD_PERM_MAX = 0xFF,
  ROW_SHL1 = 0x101,
  ROW_SHL_MAX = 0x10F,
  ROW_SHR1 = 0x111,
  ROW_SHR_MAX = 0x11F,
  ROW_ROR1 = 0x121,
  ROW_ROR_MAX = 0x12F,
  WF_SHL1 = 0x130,
  WF_ROL1 = 0x134,
  WF_SRL1 = 0x138,
  WF_ROR1 = 0x13C,
  ROW_MIRROR = 0x140,
  ROW_HALF_MIRROR = 0x141,
  ROW_BCAST15 = 0x142,
  ROW_BCAST31 = 0x143,
  ROW_SELECT_BASE = 0x150,
  ROW_SELECT_MAX = 0x15F,
  // MI400 names this range DPP_ROW_SHARE. Keep aliases for ISA-specific code.
  ROW_SHARE_BASE = ROW_SELECT_BASE,
  ROW_SHARE_MAX = ROW_SELECT_MAX,
  ROW_XMASK_BASE = 0x160,
  ROW_XMASK_MAX = 0x16F,
};

/// @brief Return true when a DPP control can read past a row or wave edge.
///
/// These controls leave some destination lanes unwritten when BOUND_CTRL is
/// zero. Rotates, mirrors, quad permutations, row-select, and row-xmask always
/// map to valid lanes.
inline bool dpp_ctrl_produces_oob(uint32_t dpp_ctrl) {
  return (dpp_ctrl >= ROW_SHL1 && dpp_ctrl <= ROW_SHL_MAX) ||
         (dpp_ctrl >= ROW_SHR1 && dpp_ctrl <= ROW_SHR_MAX) || dpp_ctrl == WF_SHL1 ||
         dpp_ctrl == WF_SRL1 || dpp_ctrl == ROW_BCAST15 || dpp_ctrl == ROW_BCAST31;
}

/// @brief Return true for a documented DPP16 control encoding.
inline bool dpp_ctrl_is_valid(uint32_t dpp_ctrl, bool allow_wave_ops, bool allow_row_bcast,
                              bool allow_row_xmask) {
  return dpp_ctrl <= QUAD_PERM_MAX || (dpp_ctrl >= ROW_SHL1 && dpp_ctrl <= ROW_SHL_MAX) ||
         (dpp_ctrl >= ROW_SHR1 && dpp_ctrl <= ROW_SHR_MAX) ||
         (dpp_ctrl >= ROW_ROR1 && dpp_ctrl <= ROW_ROR_MAX) ||
         (allow_wave_ops && (dpp_ctrl == WF_SHL1 || dpp_ctrl == WF_ROL1 || dpp_ctrl == WF_SRL1 ||
                             dpp_ctrl == WF_ROR1)) ||
         dpp_ctrl == ROW_MIRROR || dpp_ctrl == ROW_HALF_MIRROR ||
         (allow_row_bcast && (dpp_ctrl == ROW_BCAST15 || dpp_ctrl == ROW_BCAST31)) ||
         (dpp_ctrl >= ROW_SELECT_BASE && dpp_ctrl <= ROW_SELECT_MAX) ||
         (allow_row_xmask && dpp_ctrl >= ROW_XMASK_BASE && dpp_ctrl <= ROW_XMASK_MAX);
}

inline bool is_src_dpp8(uint32_t src0) { return src0 == SRC_DPP8_FI_0 || src0 == SRC_DPP8_FI_1; }

inline uint32_t src_dpp8_fi(uint32_t src0) { return src0 == SRC_DPP8_FI_1 ? 1u : 0u; }

/// @brief Add the lane selectors for a DPP8 instruction.
inline void append_dpp8_disassembly(std::string &out, uint32_t lane_sel, uint32_t fi) {
  out += " dpp8:[";
  for (uint32_t lane = 0; lane < 8; ++lane) {
    if (lane != 0)
      out += ',';
    out += std::to_string((lane_sel >> (lane * 3)) & 0x7);
  }
  out += ']';
  if (fi != 0)
    out += " fi:1";
}

} // namespace dpp

namespace sdwa {

/// @brief SDWA sub-dword selection values stored by VOP encoding models.
enum SdwaSel : uint32_t {
  BYTE_0 = 0,
  BYTE_1 = 1,
  BYTE_2 = 2,
  BYTE_3 = 3,
  WORD_0 = 4,
  WORD_1 = 5,
  DWORD = 6,
};

/// @brief SDWA handling for destination bits outside the selected sub-dword.
enum SdwaUnused : uint32_t {
  UNUSED_PAD = 0,
  UNUSED_SEXT = 1,
  UNUSED_PRESERVE = 2,
};

/// @brief Floating-point representation used by SDWA source modifiers.
///
/// SDWA selection and sign extension apply to every source. Absolute-value and
/// negate fields apply only to floating-point sources, and their sign bit
/// depends on the semantic source type rather than the selector width.
enum class SourceModifierFormat {
  NONE,
  F16,
  BF16,
  F32,
};

} // namespace sdwa

/// @brief Return the VOP3 output-select field across MRISA spelling variants.
template <typename MachineInst> inline uint32_t vop3_opsel(const MachineInst &inst) {
  if constexpr (requires { inst.opsel; })
    return inst.opsel;
  else if constexpr (requires { inst.op_sel; })
    return inst.op_sel;
  else
    return 0;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_INSTRUCTION_ENCODING_H_
