// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file legalization/gfx1250_b0_to_a0.cpp
/// @brief Handwritten gfx1250 B0-to-A0 legalization classification.

#include "rocjitsu/code/dbt/legalization/gfx1250_b0_to_a0.h"

#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/semantic/gfx1250_flat_scratch_base.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/instruction.h"

#include <array>
#include <cstring>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocjitsu {
namespace {

// gfx1250 is Wave32-only.
constexpr uint8_t kGfx1250WaveSize =
    static_cast<uint8_t>(isa_properties(ROCJITSU_CODE_ARCH_CDNA5).wave_size_max);

/// @brief Instructions that require a B0-to-A0 expansion or rejection.
/// @details These operation groups are intentionally broader than the implemented
/// rewrite table. Unimplemented siblings must fail closed. Opcode-keyed spacing
/// rewrites need no entry here because they retain the ordinary copy fallback.
[[nodiscard]] bool requires_b0_to_a0_expansion(const Instruction &inst) {
  if (inst.amdgpu_memory_addressing() == amdgpu::MemoryAddressing::Cluster)
    return true;
  switch (inst.amdgpu_operation()) {
  // Relocation changes clause membership and placement constraints.
  case amdgpu::Operation::S_CLAUSE:
  // Exact operations implemented by this expansion profile.
  case amdgpu::Operation::DS_LOAD_2ADDR_B32:
  case amdgpu::Operation::DS_LOAD_2ADDR_B64:
  case amdgpu::Operation::DS_LOAD_2ADDR_STRIDE64_B32:
  case amdgpu::Operation::DS_LOAD_2ADDR_STRIDE64_B64:
  case amdgpu::Operation::DS_STORE_2ADDR_B32:
  case amdgpu::Operation::DS_STORE_2ADDR_B64:
  case amdgpu::Operation::DS_STORE_2ADDR_STRIDE64_B32:
  case amdgpu::Operation::DS_STORE_2ADDR_STRIDE64_B64:
  case amdgpu::Operation::DS_STOREXCHG_2ADDR_RTN_B32:
  case amdgpu::Operation::DS_STOREXCHG_2ADDR_RTN_B64:
  case amdgpu::Operation::DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32:
  case amdgpu::Operation::DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64:
  case amdgpu::Operation::DS_LOAD_ADDTID_B32:
  case amdgpu::Operation::DS_STORE_ADDTID_B32:
  case amdgpu::Operation::V_CVT_PK_FP8_F32:
  case amdgpu::Operation::V_CVT_SR_FP8_F32:
  case amdgpu::Operation::TENSOR_LOAD_TO_LDS:
  // CLAMP selects the B0-only conversion mode; checked by the caller.
  case amdgpu::Operation::V_CVT_F32_FP8:
  // B0-only dense K=128 forms and low-precision scale-prefix rules.
  case amdgpu::Operation::V_WMMA_F16_16X16X128_BF8_BF8:
  case amdgpu::Operation::V_WMMA_F16_16X16X128_BF8_FP8:
  case amdgpu::Operation::V_WMMA_F16_16X16X128_FP8_BF8:
  case amdgpu::Operation::V_WMMA_F16_16X16X128_FP8_FP8:
  case amdgpu::Operation::V_WMMA_F32_16X16X128_BF8_BF8:
  case amdgpu::Operation::V_WMMA_F32_16X16X128_BF8_FP8:
  case amdgpu::Operation::V_WMMA_F32_16X16X128_FP8_BF8:
  case amdgpu::Operation::V_WMMA_F32_16X16X128_FP8_FP8:
  case amdgpu::Operation::V_WMMA_F32_32X16X128_F4:
  case amdgpu::Operation::V_WMMA_F32_16X16X128_F8F6F4:
  case amdgpu::Operation::V_WMMA_SCALE16_F32_16X16X128_F8F6F4:
  case amdgpu::Operation::V_WMMA_SCALE16_F32_32X16X128_F4:
  case amdgpu::Operation::V_WMMA_SCALE_F32_16X16X128_F8F6F4:
  case amdgpu::Operation::V_WMMA_SCALE_F32_32X16X128_F4:
  // Integer co-execution spacing is longer on A0. Unimplemented shapes
  // remain rejected until their CFG-aware spacing rule is available.
  case amdgpu::Operation::V_SWMMAC_I32_16X16X128_IU8:
  case amdgpu::Operation::V_SWMMAC_I32_16X16X32_IU4:
  case amdgpu::Operation::V_SWMMAC_I32_16X16X32_IU8:
  case amdgpu::Operation::V_SWMMAC_I32_16X16X64_IU4:
  case amdgpu::Operation::V_WMMA_I32_16X16X16_IU4:
  case amdgpu::Operation::V_WMMA_I32_16X16X16_IU8:
  case amdgpu::Operation::V_WMMA_I32_16X16X32_IU4:
  case amdgpu::Operation::V_WMMA_I32_16X16X64_IU8:
    return true;
  default:
    return false;
  }
}

/// @brief True when a B0 FP8 conversion selects the B0-only E5M3 mode.
///
/// @details The affected VOP3 conversions reuse CLAMP as the E5M3 selector on
/// B0. A0 implements the same CLAMP=0 E4M3 operation, so those instructions
/// must remain on the ordinary byte-copy path. CLAMP lives in the eight-byte
/// VOP3 base encoding; a trailing literal increases the decoded size without
/// moving that field.
[[nodiscard]] bool requires_fp8_clamp_emulation(const Instruction &inst) {
  const amdgpu::Operation operation = inst.amdgpu_operation();
  const bool affected = operation == amdgpu::Operation::V_CVT_PK_FP8_F32 ||
                        operation == amdgpu::Operation::V_CVT_SR_FP8_F32 ||
                        operation == amdgpu::Operation::V_CVT_F32_FP8;
  const bool is_vop3 = inst.encoding_id() >= cdna5::encoding::kVop3 &&
                       inst.encoding_id() <= cdna5::encoding::kVop3OpHi6;
  if (!affected || !is_vop3 || inst.size() < static_cast<int>(sizeof(cdna5::Vop3MachineInst)) ||
      inst.raw_encoding() == nullptr)
    return false;

  cdna5::Vop3MachineInst encoding{};
  std::memcpy(&encoding, inst.raw_encoding(), sizeof(encoding));
  return encoding.clamp != 0;
}

[[nodiscard]] bool is_wmma_completion_wait(const Instruction &inst) {
  if (inst.size() != static_cast<int>(sizeof(uint32_t)) || inst.raw_encoding() == nullptr ||
      inst.amdgpu_operation() != amdgpu::Operation::S_WAIT_ALU)
    return false;

  // S_WAIT_ALU SIMM16[15:12] is VA_VDST. Zero waits for all outstanding
  // destination writes, regardless of any additional counters the source
  // instruction also drains.
  constexpr uint32_t kVaVdstMask = 0xf000u;
  return (inst.raw_encoding()[0] & kVaVdstMask) == 0;
}

[[nodiscard]] bool has_wmma_completion_wait_ahead(
    const Instruction &inst,
    const std::unordered_map<uint64_t, const Instruction *> &source_instruction_by_offset,
    const Gfx1250VgprMsbAnalysis &vgpr_msb) {
  RegisterSet pending_defs =
      InstDefUse(inst, kGfx1250WaveSize, &vgpr_msb, UnknownVgprDefPolicy::ExpandAll)
          .defs.ordinary_only();
  uint64_t next_offset = inst.src_loc() + inst.size();
  while (true) {
    const auto next_it = source_instruction_by_offset.find(next_offset);
    if (next_it == source_instruction_by_offset.end())
      return false;

    const Instruction &next = *next_it->second;
    constexpr uint64_t kControlTransferOrTerminator =
        BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR;
    if ((next.flags() & kControlTransferOrTerminator) != 0)
      return false;
    if (is_wmma_completion_wait(next))
      return true;

    // Bound the scan to affected WMMA and the exact scalar instructions that
    // generated split forms may place before their trailing completion wait.
    // The def/use check below independently verifies VGPR transparency for
    // both generated and hand-written source streams.
    const amdgpu::Operation operation = next.amdgpu_operation();
    const bool canonical_split_scaffolding = operation == amdgpu::Operation::S_WAIT_XCNT ||
                                             operation == amdgpu::Operation::S_SET_VGPR_MSB;
    if (!canonical_split_scaffolding && !gfx1250_b0_to_a0_requires_wmma_completion_wait(next))
      return false;

    const InstDefUse access(next, kGfx1250WaveSize, &vgpr_msb, UnknownVgprDefPolicy::ExpandAll);
    if (access.uses.intersects(pending_defs))
      return false;
    pending_defs |= access.defs.ordinary_only();

    if (next.size() <= 0)
      return false;
    next_offset += static_cast<uint64_t>(next.size());
  }
}

} // namespace

bool gfx1250_b0_to_a0_requires_wmma_completion_wait(const Instruction &inst) {
  switch (inst.amdgpu_operation()) {
  case amdgpu::Operation::V_WMMA_F16_16X16X128_BF8_BF8:
  case amdgpu::Operation::V_WMMA_F16_16X16X128_BF8_FP8:
  case amdgpu::Operation::V_WMMA_F16_16X16X128_FP8_BF8:
  case amdgpu::Operation::V_WMMA_F16_16X16X128_FP8_FP8:
  case amdgpu::Operation::V_WMMA_F16_16X16X64_BF8_BF8:
  case amdgpu::Operation::V_WMMA_F16_16X16X64_BF8_FP8:
  case amdgpu::Operation::V_WMMA_F16_16X16X64_FP8_BF8:
  case amdgpu::Operation::V_WMMA_F16_16X16X64_FP8_FP8:
  case amdgpu::Operation::V_WMMA_F32_16X16X128_BF8_BF8:
  case amdgpu::Operation::V_WMMA_F32_16X16X128_BF8_FP8:
  case amdgpu::Operation::V_WMMA_F32_16X16X128_F8F6F4:
  case amdgpu::Operation::V_WMMA_F32_16X16X128_FP8_BF8:
  case amdgpu::Operation::V_WMMA_F32_16X16X128_FP8_FP8:
  case amdgpu::Operation::V_WMMA_F32_16X16X16_BF8_BF8:
  case amdgpu::Operation::V_WMMA_F32_16X16X16_BF8_FP8:
  case amdgpu::Operation::V_WMMA_F32_16X16X16_FP8_BF8:
  case amdgpu::Operation::V_WMMA_F32_16X16X16_FP8_FP8:
  case amdgpu::Operation::V_WMMA_F32_16X16X64_BF8_BF8:
  case amdgpu::Operation::V_WMMA_F32_16X16X64_BF8_FP8:
  case amdgpu::Operation::V_WMMA_F32_16X16X64_FP8_BF8:
  case amdgpu::Operation::V_WMMA_F32_16X16X64_FP8_FP8:
  case amdgpu::Operation::V_WMMA_F32_32X16X128_F4:
  case amdgpu::Operation::V_WMMA_SCALE16_F32_16X16X128_F8F6F4:
  case amdgpu::Operation::V_WMMA_SCALE16_F32_32X16X128_F4:
  case amdgpu::Operation::V_WMMA_SCALE_F32_16X16X128_F8F6F4:
  case amdgpu::Operation::V_WMMA_SCALE_F32_32X16X128_F4:
    return true;
  default:
    return false;
  }
}

void gfx1250_b0_to_a0_append_wmma_completion_wait_if_needed(
    const Instruction &inst,
    const std::unordered_map<uint64_t, const Instruction *> &source_instruction_by_offset,
    const Gfx1250VgprMsbAnalysis &vgpr_msb, std::vector<uint32_t> &words) {
  if (!gfx1250_b0_to_a0_requires_wmma_completion_wait(inst) ||
      has_wmma_completion_wait_ahead(inst, source_instruction_by_offset, vgpr_msb)) {
    return;
  }

  // TODO: Replace this conservative producer-side drain with block-local
  // S_WAIT_ALU analysis over the final translated instruction stream. It should
  // track VA_VDST/VM_VSRC dependencies and wait only before a dependent use.
  // B0 code may schedule scaled WMMA consumers under SCHED_MODE 2 using B0
  // completion timing. A0 has a lower FP8/FP4 WMMA issue rate, and mode 2 makes
  // software responsible for VA_VDST dependencies. Wait for pending VALU
  // destinations without draining unrelated ALU dependency counters.
  // The no-wait default is 0xff9f; clearing only VA_VDST[15:12] gives 0x0f9f.
  constexpr uint16_t kWaitVaVdstZero = 0x0f9f;
  words.push_back(cdna5::build_sopp(cdna5::kSWaitAluSopp, {.simm16 = kWaitVaVdstZero})[0]);
}

bool gfx1250_b0_to_a0_is_deferred_family(const Instruction &inst) {
  // s_sleep and s_sleep_var are deliberately absent. They behave identically on
  // A0 and B0. Only s_monitor_sleep('forever') with MWAIT=0 requires an A0
  // translation. Copying a plain sleep through is the correct translation, not
  // an unimplemented one, so reporting
  // it said nothing and buried the reports that do name a real gap -- one RCCL
  // all_reduce run emitted 104,831 of them.
  return inst.amdgpu_operation() == amdgpu::Operation::S_GET_BARRIER_STATE ||
         inst.amdgpu_operation() == amdgpu::Operation::S_MONITOR_SLEEP;
}

const InstructionLegalization *gfx1250_b0_to_a0_legalization(const Instruction &inst) {
  // CLAMP=0 is the common E4M3 operation on both steppings. CLAMP=1 selects
  // the B0-only E5M3 behavior and therefore requires a semantic expansion.
  const amdgpu::Operation operation = inst.amdgpu_operation();
  const bool fp8_clamp_family = operation == amdgpu::Operation::V_CVT_PK_FP8_F32 ||
                                operation == amdgpu::Operation::V_CVT_SR_FP8_F32 ||
                                operation == amdgpu::Operation::V_CVT_F32_FP8;
  if (fp8_clamp_family && !requires_fp8_clamp_emulation(inst))
    return nullptr;

  if (!requires_b0_to_a0_expansion(inst)) {
    return nullptr;
  }

  // The runtime uses only the action and target opcode for this revision-specific
  // classification. Source keys remain zero because matching is performed on
  // the fully decoded operation, which is necessary for contextual gfx1250
  // variants that share structural opcode fields.
  static constexpr InstructionLegalization kExpand{
      .src_opcode = 0,
      .src_encoding_id = 0,
      .action = Action::Expand,
      .target_opcode = 0,
  };
  return &kExpand;
}

} // namespace rocjitsu
