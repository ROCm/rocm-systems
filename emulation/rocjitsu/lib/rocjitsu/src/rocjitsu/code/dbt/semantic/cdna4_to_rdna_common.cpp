// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna4_to_rdna_common.cpp
/// @brief Lowerings shared by CDNA4-to-RDNA semantic rule tables.

#include "rocjitsu/code/dbt/semantic/cdna4_to_rdna_common.h"

#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/opcodes.h"
#include "rocjitsu/isa/instruction.h"

#include <array>
#include <cstring>
#include <string>
#include <utility>

namespace rocjitsu {
namespace {

/// @brief Build the explicit-SDST VOP3 form used by the carry chain.
/// @details RDNA3 and RDNA4 currently share the layout, but selecting the
/// generated builder by host ISA prevents a future XML change from silently
/// encoding one target with the other target's format.
[[nodiscard]] std::array<uint32_t, 2> build_rdna_vop3_sdst(rj_code_arch_t host_arch, uint16_t op,
                                                           uint8_t vdst, uint8_t sdst,
                                                           uint16_t src0, uint16_t src1 = 0,
                                                           uint16_t src2 = 0) {
  if (host_arch == ROCJITSU_CODE_ARCH_RDNA3) {
    return rdna3::build_vop3_sdst_enc(
        op, {.vdst = vdst, .sdst = sdst, .src0 = src0, .src1 = src1, .src2 = src2});
  }
  return rdna4::build_vop3_sdst_enc(
      op, {.vdst = vdst, .sdst = sdst, .src0 = src0, .src1 = src1, .src2 = src2});
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
  // These generated opcodes currently match on RDNA3 and RDNA4. Keep the
  // architecture-specific selection explicit so an ISA XML change cannot
  // silently make this shared lowering emit the other target's opcode.
  const uint16_t add_co_u32_op = host_arch == ROCJITSU_CODE_ARCH_RDNA3
                                     ? rdna3::kVAddCoU32Vop3SdstEnc
                                     : rdna4::kVAddCoU32Vop3SdstEnc;
  const uint16_t add_co_ci_u32_op = host_arch == ROCJITSU_CODE_ARCH_RDNA3
                                        ? rdna3::kVAddCoCiU32Vop3SdstEnc
                                        : rdna4::kVAddCoCiU32Vop3SdstEnc;

  // This lowering introduces VCC as the explicit carry register. VCC is special
  // scalar state, not a liveness-allocated scratch SGPR pair, so it must not
  // grow the ordinary SGPR descriptor allocation. Treating it as normal SGPRs
  // makes valid RDNA targets look unsupported once diagnostics become fatal.

  std::vector<uint32_t> words;

  // v_add_co_u32 writes VCC, then v_add_co_ci_u32 consumes VCC as carry-in.
  {
    auto [w0, w1] = build_rdna_vop3_sdst(host_arch, add_co_u32_op, static_cast<uint8_t>(vdst),
                                         kVccLo, src0, src2);
    words.push_back(w0);
    words.push_back(w1);
  }

  // GFX12 requires an explicit VALU dependency wait before the carry read.
  // GFX11 handles this pair through hardware scoreboarding, and SOPP opcode 8
  // is not s_wait_alu there, so only emit it for RDNA4.
  if (host_arch == ROCJITSU_CODE_ARCH_RDNA4)
    words.push_back(rdna4::build_sopp(rdna4::kSWaitAlu, {.simm16 = 0xFFFD})[0]);

  {
    auto [w0, w1] = build_rdna_vop3_sdst(
        host_arch, add_co_ci_u32_op, static_cast<uint8_t>(vdst + 1), kVccLo,
        static_cast<uint16_t>(src0 + 1), static_cast<uint16_t>(src2 + 1), kVccLo);
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
