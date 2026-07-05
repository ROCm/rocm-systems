// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna4_to_rdna_common.cpp
/// @brief Lowerings shared by CDNA4-to-RDNA semantic rule tables.

#include "rocjitsu/code/dbt/semantic/cdna4_to_rdna_common.h"

#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/instruction.h"

#include <cstring>
#include <string>
#include <utility>

namespace rocjitsu {
namespace {

/// @brief Build an RDNA VOP3 instruction word pair.
/// @details This helper is intentionally narrow: the v_lshl_add_u64 lowering
/// only needs the VOP3 form used by v_add_co_u32/v_add_co_ci_u32 on RDNA3/4.
[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_rdna_vop3(uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1 = 0, uint16_t src2 = 0) {
  const uint32_t w0 = (vdst & 0xFFu) | ((op & 0x3FFu) << 16) | (0x35u << 26);
  const uint32_t w1 = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18);
  return {w0, w1};
}

ExpandResult lower_v_lshl_add_u64(const Instruction &inst, rj_code_arch_t host_arch) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " cannot lower v_lshl_add_u64 without a complete 64-bit raw "
                                "encoding");

  cdna4::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const uint16_t vdst = src.vdst;
  const uint16_t src0 = src.src0;
  const uint16_t src2 = src.src2;

  if (host_arch != ROCJITSU_CODE_ARCH_RDNA3 && host_arch != ROCJITSU_CODE_ARCH_RDNA4)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " v_lshl_add_u64 lowering only supports RDNA3/RDNA4 hosts");
  if (vdst == 255)
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                " v_lshl_add_u64 lowering needs two destination VGPRs but vdst "
                                "is v255");

  constexpr uint16_t kVccLo = 106;
  constexpr uint16_t kOpAddCoU32 = 768;
  constexpr uint16_t kOpAddCoCiU32 = 288;
  constexpr uint8_t kSoppWaitAlu = 8;

  // This lowering introduces VCC as the explicit carry register. VCC is special
  // scalar state, not a liveness-allocated scratch SGPR pair, so it must not
  // grow the ordinary SGPR descriptor allocation. Treating it as normal SGPRs
  // makes valid RDNA targets look unsupported once diagnostics become fatal.

  std::vector<uint32_t> words;

  // v_add_co_u32 writes VCC, then v_add_co_ci_u32 consumes VCC as carry-in.
  {
    auto [w0, w1] = build_rdna_vop3(kOpAddCoU32, static_cast<uint8_t>(vdst), src0, src2);
    w0 |= (kVccLo << 8); // sdst = vcc_lo
    words.push_back(w0);
    words.push_back(w1);
  }

  // GFX12 requires an explicit VALU dependency wait before the carry read.
  // GFX11 handles this pair through hardware scoreboarding, and SOPP opcode 8
  // is not s_wait_alu there, so only emit it for RDNA4.
  if (host_arch == ROCJITSU_CODE_ARCH_RDNA4)
    words.push_back(pack_sopp(kSoppWaitAlu, 0xFFFD));

  {
    auto [w0, w1] =
        build_rdna_vop3(kOpAddCoCiU32, static_cast<uint8_t>(vdst + 1),
                        static_cast<uint16_t>(src0 + 1), static_cast<uint16_t>(src2 + 1), kVccLo);
    w0 |= (kVccLo << 8); // sdst = vcc_lo
    words.push_back(w0);
    words.push_back(w1);
  }

  return ExpandResult::success(std::move(words));
}

} // namespace

ExpandResult expand_cdna4_v_lshl_add_u64_for_rdna(const Instruction &inst, uint32_t host_arch,
                                                  uint64_t, const LivenessAnalysis &,
                                                  TranslationContext &, const LaneLayout *,
                                                  const LaneLayout *) {
  return lower_v_lshl_add_u64(inst, static_cast<rj_code_arch_t>(host_arch));
}

} // namespace rocjitsu
