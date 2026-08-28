// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instruction_encoding.h
/// @brief Lightweight AMDGPU instruction-encoding helpers.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_INSTRUCTION_ENCODING_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_INSTRUCTION_ENCODING_H_

#include <cstdint>

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
  ROW_SHARE_BASE = 0x150,
  ROW_SHARE_MAX = 0x15F,
  ROW_XMASK_BASE = 0x160,
  ROW_XMASK_MAX = 0x16F,
};

/// @brief Return true when a DPP control can read past a row or wave edge.
///
/// These controls leave some destination lanes unwritten when BOUND_CTRL is
/// zero. Rotates, mirrors, quad permutations, row-share, and row-xmask always
/// map to valid lanes.
inline bool dpp_ctrl_produces_oob(uint32_t dpp_ctrl) {
  return (dpp_ctrl >= ROW_SHL1 && dpp_ctrl <= ROW_SHL_MAX) ||
         (dpp_ctrl >= ROW_SHR1 && dpp_ctrl <= ROW_SHR_MAX) || dpp_ctrl == WF_SHL1 ||
         dpp_ctrl == WF_SRL1 || dpp_ctrl == ROW_BCAST15 || dpp_ctrl == ROW_BCAST31;
}

inline bool is_src_dpp8(uint32_t src0) { return src0 == SRC_DPP8_FI_0 || src0 == SRC_DPP8_FI_1; }

inline uint32_t src_dpp8_fi(uint32_t src0) { return src0 == SRC_DPP8_FI_1 ? 1u : 0u; }

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
