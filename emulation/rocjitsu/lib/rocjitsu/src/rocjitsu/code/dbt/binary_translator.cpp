// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/dbt/semantic_translator.h"
#include "rocjitsu/code/dbt/virtual_lds_metadata.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

inline constexpr uint64_t kKernargPreloadSkipBytes = 256;
inline constexpr uint16_t kCdna4DsEncodingFamily = 0x1B0;
inline constexpr uint16_t kCdna4EncodingFamilyMask = 0x1F8;
inline constexpr uint8_t kCdna3FlatOpLoadUbyte = 16;
inline constexpr uint8_t kCdna3FlatOpLoadSbyte = 17;
inline constexpr uint8_t kCdna3FlatOpLoadUshort = 18;
inline constexpr uint8_t kCdna3FlatOpLoadDword = 20;
inline constexpr uint8_t kCdna3FlatOpStoreByte = 24;
inline constexpr uint8_t kCdna3FlatOpStoreShort = 26;
inline constexpr uint8_t kCdna3FlatOpStoreDword = 28;
inline constexpr uint8_t kCdna3FlatOpLoadShortD16 = 36;
inline constexpr uint8_t kCdna3FlatOpLoadShortD16Hi = 37;
inline constexpr uint8_t kCdna3FlatOpStoreShortD16Hi = 27;
inline constexpr uint8_t kCdna3Vop2OpAddU32 = 52;
inline constexpr uint8_t kCdna3Vop1OpMovB32 = 1;
inline constexpr uint8_t kCdna3Vop1OpReadfirstlaneB32 = 2;
inline constexpr uint8_t kCdna3Sop2OpAddU32 = 0;
inline constexpr uint8_t kCdna3Sop2OpAddcU32 = 4;
inline constexpr uint8_t kCdna3Sop2OpMulI32 = 36;
inline constexpr uint8_t kCdna3SoppOpWaitcnt = 12;
inline constexpr uint8_t kCdna3SoppOpCbranchExecz = 8;
inline constexpr uint16_t kCdnaWaitcntAll0 = 0x0000;
inline constexpr uint32_t kFlatGlobalPositiveImm13Max = 4095;
inline constexpr uint32_t kCdnaOrdinarySgprLimit = 102;
inline constexpr uint32_t kCdnaSmemImmediateByteOffsetMax = 0x1FFFFF;
inline constexpr uint32_t kVirtualLdsStateBackingBaseOffset = 0;
inline constexpr uint32_t kVirtualLdsStateStrideXOffset = 8;
inline constexpr uint32_t kVirtualLdsStateStrideYOffset = 12;
inline constexpr uint32_t kVirtualLdsStateStrideZOffset = 16;
inline constexpr uint32_t kVirtualLdsRuntimeStateBytes = 24;
inline constexpr uint32_t kCdna3MaxHardwareLdsBytes = 64 * 1024;
inline constexpr uint64_t kDirectBranchIslandSpacingBytes = 64 * 1024;
inline constexpr uint16_t kDirectBranchIslandPoolSlots = 16;

struct VirtualLdsVgprRange {
  uint16_t base = 0;
  uint16_t count = 0;
};

struct VirtualLdsAddressTemp {
  uint8_t reg = 0;
  bool spilled = false;
  uint32_t spill_offset = 0;
};

struct VirtualLdsBaseSgprReservation {
  uint16_t base = 0;
  uint16_t prologue_temp = 0;
  bool spill_per_use = false;
};

EncodingTranslateFn select_encoding_translator(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3)
    return cdna4_to_cdna3::translate_encoding_cdna4_to_cdna3;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    return cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3;
  return nullptr;
}

LegalizationLookupFn select_legalization(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna4, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_cdna3, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna3, enc_id, opcode);
    };
  }
  return nullptr;
}

[[nodiscard]] std::vector<uint32_t> raw_words_for_inst(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return {};
  return {raw, raw + inst.size() / sizeof(uint32_t)};
}

[[nodiscard]] uint32_t text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  uint32_t word = 0;
  if (offset + sizeof(word) <= text.size())
    std::memcpy(&word, text.data() + offset, sizeof(word));
  return word;
}

[[nodiscard]] bool words_changed(std::span<const uint32_t> before,
                                 std::span<const uint32_t> after) {
  if (before.size() != after.size())
    return true;
  return !std::ranges::equal(before, after);
}

void append_diagnostic(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticSeverity severity,
                       DiagnosticKind kind, std::string message,
                       std::optional<uint64_t> guest_offset = std::nullopt,
                       std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  diagnostics.push_back({.severity = severity,
                         .kind = kind,
                         .guest_offset = guest_offset,
                         .mnemonic = std::move(mnemonic),
                         .message = std::move(message),
                         .required_work = std::move(required_work)});
}

void append_error(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticKind kind,
                  std::string message, std::optional<uint64_t> guest_offset = std::nullopt,
                  std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  append_diagnostic(diagnostics, DiagnosticSeverity::Error, kind, std::move(message), guest_offset,
                    std::move(mnemonic), std::move(required_work));
}

void append_warning(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticKind kind,
                    std::string message, std::optional<uint64_t> guest_offset = std::nullopt,
                    std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  append_diagnostic(diagnostics, DiagnosticSeverity::Warning, kind, std::move(message),
                    guest_offset, std::move(mnemonic), std::move(required_work));
}

void append_diagnostics(std::vector<TranslationDiagnostic> &dst,
                        const std::vector<TranslationDiagnostic> &src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

[[nodiscard]] std::string kernel_label(const KdTranslation &translation) {
  if (!translation.kernel_name.empty())
    return translation.kernel_name;

  std::ostringstream os;
  os << ".text+0x" << std::hex << translation.entry_text_offset;
  return os.str();
}

[[nodiscard]] bool is_cdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

[[nodiscard]] bool supports_direct_branch_island_window(const Instruction &inst) {
  if (inst.mnemonic() == "s_branch")
    return true;

  // Conditional branches need a two-word source window: invert the condition to
  // skip over an unconditional branch into the island chain. Keep the list
  // matched to build_inverted_conditional_skip() in the patch layer so a window
  // is reserved only for branch forms the patcher can actually lower.
  return inst.mnemonic() == "s_cbranch_scc0" || inst.mnemonic() == "s_cbranch_scc1" ||
         inst.mnemonic() == "s_cbranch_vccz" || inst.mnemonic() == "s_cbranch_vccnz" ||
         inst.mnemonic() == "s_cbranch_execz" || inst.mnemonic() == "s_cbranch_execnz";
}

void append_direct_branch_island_pool(std::vector<uint8_t> &kernel_text, KernelTextLayout &layout,
                                      rj_code_arch_t arch) {
  const uint64_t skip_offset = kernel_text.size();
  const uint32_t skip_pool =
      build_s_branch(static_cast<int16_t>(kDirectBranchIslandPoolSlots), arch);
  append_words(kernel_text, std::span<const uint32_t>(&skip_pool, 1));

  // Normal fallthrough executes the skip above and lands after the pool. A
  // patched out-of-range branch may instead target one of these private slots,
  // each of which is later rewritten to an unconditional branch to the next
  // island or final target.
  const uint32_t placeholder = build_s_branch(0, arch);
  for (uint16_t i = 0; i < kDirectBranchIslandPoolSlots; ++i) {
    layout.branch_island_slots.push_back(skip_offset + sizeof(uint32_t) +
                                         static_cast<uint64_t>(i) * sizeof(uint32_t));
    append_words(kernel_text, std::span<const uint32_t>(&placeholder, 1));
  }
}

[[nodiscard]] bool virtual_lds_vgpr_ranges_overlap(uint16_t lhs_base, uint16_t lhs_count,
                                                   uint16_t rhs_base, uint16_t rhs_count) {
  return lhs_base < rhs_base + rhs_count && rhs_base < lhs_base + lhs_count;
}

[[nodiscard]] bool virtual_lds_vgpr_is_forbidden(uint16_t reg,
                                                 std::span<const VirtualLdsVgprRange> forbidden) {
  return std::ranges::any_of(forbidden, [reg](const VirtualLdsVgprRange &range) {
    return virtual_lds_vgpr_ranges_overlap(reg, 1, range.base, range.count);
  });
}

[[nodiscard]] std::optional<VirtualLdsAddressTemp>
choose_virtual_lds_address_temp(TranslationContext &context,
                                std::span<const VirtualLdsVgprRange> forbidden) {
  // Do not ask liveness for a "free" register here. Generic virtual-LDS lowering
  // runs after semantic lowering and must be correct even when the kernel's VGPRs
  // are all live across the memory operation. Pick an ordinary VGPR that the DS
  // instruction itself is not reading/writing, spill it to the per-lane private
  // segment, and restore it before the replacement sequence finishes.
  const uint32_t initial_vgprs = std::min<uint32_t>(context.num_vgprs, 256);
  for (uint32_t reg = initial_vgprs; reg > 0; --reg) {
    const uint16_t candidate = static_cast<uint16_t>(reg - 1);
    if (virtual_lds_vgpr_is_forbidden(candidate, forbidden))
      continue;
    return VirtualLdsAddressTemp{
        .reg = static_cast<uint8_t>(candidate),
        .spilled = true,
    };
  }

  // Extremely small test kernels can name every allocated VGPR in the DS
  // instruction. In that case grow the descriptor by one scratch VGPR instead of
  // failing; this path is not expected for real fp16 kernels, but it keeps the
  // lowering well-defined for minimal fixtures.
  const uint32_t next_vgpr = std::max(context.num_vgprs, context.required_vgpr_count);
  if (next_vgpr < 256) {
    context.require_vgprs(next_vgpr + 1);
    return VirtualLdsAddressTemp{.reg = static_cast<uint8_t>(next_vgpr)};
  }

  return std::nullopt;
}

uint32_t assign_virtual_lds_spill_offsets(TranslationContext &context,
                                          std::initializer_list<VirtualLdsAddressTemp *> temps,
                                          uint32_t extra_dwords = 0) {
  uint32_t spilled_count = 0;
  for (const VirtualLdsAddressTemp *temp : temps) {
    if (temp != nullptr && temp->spilled)
      ++spilled_count;
  }
  const uint32_t total_dwords = spilled_count + extra_dwords;
  if (total_dwords == 0)
    return 0;

  // Spill slots are reusable across lowering sites, but simultaneous temps in a
  // single replacement sequence must occupy distinct dwords. Reserve one window
  // for this sequence and assign each spilled VGPR a unique offset inside it.
  const uint32_t base_offset = context.reserve_semantic_spill_dwords(total_dwords);
  uint32_t index = 0;
  for (VirtualLdsAddressTemp *temp : temps) {
    if (temp == nullptr || !temp->spilled)
      continue;
    temp->spill_offset = base_offset + index * sizeof(uint32_t);
    ++index;
  }
  return base_offset + spilled_count * sizeof(uint32_t);
}

void emit_cdna3_scratch_store_b32(std::vector<uint32_t> &words, uint8_t data, uint32_t byte_offset);
void emit_cdna3_scratch_load_b32(std::vector<uint32_t> &words, uint8_t dst, uint32_t byte_offset);
[[nodiscard]] uint32_t build_cdna3_v_mov_b32(uint8_t vdst, uint16_t src0);
[[nodiscard]] uint32_t build_cdna3_v_readfirstlane_b32(uint8_t sdst, uint8_t vsrc);
[[nodiscard]] uint32_t build_cdna3_s_cbranch_execz(uint16_t simm16);
[[nodiscard]] uint32_t build_cdna3_s_add_u32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1);
[[nodiscard]] uint32_t build_cdna3_s_addc_u32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1);
[[nodiscard]] uint32_t build_cdna3_s_mul_i32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1);

[[nodiscard]] std::optional<std::array<VirtualLdsAddressTemp, 2>>
choose_virtual_lds_base_spill_temps(TranslationContext &context,
                                    std::vector<VirtualLdsVgprRange> &forbidden) {
  std::array<VirtualLdsAddressTemp, 2> temps{};
  for (VirtualLdsAddressTemp &temp : temps) {
    auto chosen = choose_virtual_lds_address_temp(context, forbidden);
    if (!chosen)
      return std::nullopt;
    temp = *chosen;
    forbidden.push_back({.base = temp.reg, .count = 1});
  }
  return temps;
}

void emit_virtual_lds_temp_spill_stores(std::vector<uint32_t> &words,
                                        std::initializer_list<VirtualLdsAddressTemp *> temps) {
  bool emitted = false;
  for (VirtualLdsAddressTemp *temp : temps) {
    if (temp == nullptr || !temp->spilled)
      continue;
    emit_cdna3_scratch_store_b32(words, temp->reg, temp->spill_offset);
    emitted = true;
  }
  if (emitted)
    words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntAll0));
}

void emit_virtual_lds_temp_spill_loads(std::vector<uint32_t> &words,
                                       std::initializer_list<VirtualLdsAddressTemp *> temps) {
  bool emitted = false;
  for (VirtualLdsAddressTemp *temp : temps) {
    if (temp == nullptr || !temp->spilled)
      continue;
    emit_cdna3_scratch_load_b32(words, temp->reg, temp->spill_offset);
    emitted = true;
  }
  if (emitted)
    words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntAll0));
}

void emit_virtual_lds_base_spill_setup(std::vector<uint32_t> &words,
                                       const TranslationContext &context,
                                       const std::array<VirtualLdsAddressTemp, 2> &temps,
                                       uint32_t saved_sgpr_offset) {
  const auto base = static_cast<uint8_t>(context.virtual_lds_base_sgpr);
  words.push_back(build_cdna3_v_mov_b32(temps[0].reg, base));
  words.push_back(build_cdna3_v_mov_b32(temps[1].reg, static_cast<uint8_t>(base + 1)));
  if (context.virtual_lds_base_pointer_spilled) {
    // The descriptor-selected pointer SGPR pair is guest-owned after entry. For
    // descriptor-full kernels, the entry prologue saved the backing pointer in
    // persistent private scratch; each per-use borrow reloads it through VGPRs
    // so the original scalar pair can be restored after the virtual LDS access.
    emit_cdna3_scratch_store_b32(words, temps[0].reg, saved_sgpr_offset);
    emit_cdna3_scratch_store_b32(words, temps[1].reg, saved_sgpr_offset + sizeof(uint32_t));
    emit_cdna3_scratch_load_b32(words, temps[0].reg, context.virtual_lds_base_pointer_spill_offset);
    emit_cdna3_scratch_load_b32(words, temps[1].reg,
                                context.virtual_lds_base_pointer_spill_offset + sizeof(uint32_t));
    words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntAll0));
    words.push_back(build_cdna3_v_readfirstlane_b32(base, temps[0].reg));
    words.push_back(build_cdna3_v_readfirstlane_b32(static_cast<uint8_t>(base + 1), temps[1].reg));
    return;
  }
  auto [load0, load1] = Cdna3MemoryInstructionBuilder::smem_load_dwordx2(
      base, static_cast<uint8_t>(context.virtual_lds_kernarg_segment_ptr_sgpr),
      context.virtual_lds_kernarg_pointer_offset);
  words.push_back(load0);
  words.push_back(load1);
  words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntAll0));
}

void emit_virtual_lds_base_spill_restore(std::vector<uint32_t> &words,
                                         const TranslationContext &context,
                                         const std::array<VirtualLdsAddressTemp, 2> &temps,
                                         uint32_t saved_sgpr_offset) {
  const auto base = static_cast<uint8_t>(context.virtual_lds_base_sgpr);
  if (context.virtual_lds_base_pointer_spilled) {
    emit_cdna3_scratch_load_b32(words, temps[0].reg, saved_sgpr_offset);
    emit_cdna3_scratch_load_b32(words, temps[1].reg, saved_sgpr_offset + sizeof(uint32_t));
    words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntAll0));
  }
  words.push_back(build_cdna3_v_readfirstlane_b32(base, temps[0].reg));
  words.push_back(build_cdna3_v_readfirstlane_b32(static_cast<uint8_t>(base + 1), temps[1].reg));
}

[[nodiscard]] bool guard_virtual_lds_execz(std::vector<uint32_t> &words,
                                           const TranslationContext &context) {
  if (!context.virtual_lds_base_sgpr_spill_per_use)
    return true;
  if (words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max()))
    return false;
  words.insert(words.begin(), build_cdna3_s_cbranch_execz(static_cast<uint16_t>(words.size())));
  return true;
}

void emit_cdna3_scratch_store_b32(std::vector<uint32_t> &words, uint8_t data,
                                  uint32_t byte_offset) {
  auto [w0, w1] = Cdna3MemoryInstructionBuilder::flat_scratch_dword(kCdna3FlatOpStoreDword, data,
                                                                    byte_offset, false);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_cdna3_scratch_load_b32(std::vector<uint32_t> &words, uint8_t dst, uint32_t byte_offset) {
  auto [w0, w1] = Cdna3MemoryInstructionBuilder::flat_scratch_dword(kCdna3FlatOpLoadDword, dst,
                                                                    byte_offset, true);
  words.push_back(w0);
  words.push_back(w1);
}

void emit_cdna3_v_add_u32_literal(std::vector<uint32_t> &words, uint8_t vdst, uint8_t vsrc1,
                                  uint32_t literal) {
  auto [w0, w1] = build_cdna3_vop2_literal(kCdna3Vop2OpAddU32, vdst, vsrc1, literal);
  words.push_back(w0);
  words.push_back(w1);
}

[[nodiscard]] uint32_t build_cdna3_vop1(uint8_t op, uint8_t vdst, uint16_t src0) {
  cdna3::Vop1MachineInst dst{};
  dst.encoding = 0x3F;
  dst.op = op;
  dst.vdst = vdst;
  dst.src0 = src0 & 0x1FF;
  uint32_t word = 0;
  std::memcpy(&word, &dst, sizeof(word));
  return word;
}

[[nodiscard]] uint32_t build_cdna3_v_mov_b32(uint8_t vdst, uint16_t src0) {
  return build_cdna3_vop1(kCdna3Vop1OpMovB32, vdst, src0);
}

[[nodiscard]] uint32_t build_cdna3_v_readfirstlane_b32(uint8_t sdst, uint8_t vsrc) {
  // VOP1 SRC0 uses the scalar-source operand namespace. Plain values 0..127
  // select SGPR/special scalar operands such as EXEC_LO/EXEC_HI; VGPR operands
  // are encoded as 256 + vN. `v_readfirstlane_b32` must read the per-lane VGPR
  // temp, not one of those scalar operands, because virtual-LDS spill restore
  // uses it to rebuild the borrowed backing-pointer SGPR pair.
  return build_cdna3_vop1(kCdna3Vop1OpReadfirstlaneB32, sdst, static_cast<uint16_t>(256u + vsrc));
}

[[nodiscard]] uint32_t build_cdna3_s_cbranch_execz(uint16_t simm16) {
  return pack_sopp(kCdna3SoppOpCbranchExecz, simm16);
}

[[nodiscard]] uint32_t build_cdna3_s_add_u32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return pack_sop2(kCdna3Sop2OpAddU32, sdst, ssrc0, ssrc1);
}

[[nodiscard]] uint32_t build_cdna3_s_addc_u32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return pack_sop2(kCdna3Sop2OpAddcU32, sdst, ssrc0, ssrc1);
}

[[nodiscard]] uint32_t build_cdna3_s_mul_i32(uint8_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return pack_sop2(kCdna3Sop2OpMulI32, sdst, ssrc0, ssrc1);
}

[[nodiscard]] uint32_t max_descriptor_sgpr_allocation_for_long_branch(rj_code_arch_t arch) {
  // This mirrors the descriptor-side CDNA limit used by
  // KernelDescriptorTranslator. Long direct branches consume their scratch pair
  // at the final s_setpc_b64/s_swappc_b64 transfer, so DBT may only use a pair
  // that can be made descriptor-backed for the destination kernel.
  if (is_cdna_arch(arch))
    return 112;
  return 0;
}

[[nodiscard]] std::optional<uint16_t> next_long_branch_sgpr_pair(const TranslationContext &context,
                                                                 rj_code_arch_t arch) {
  const uint32_t current = std::max(context.num_sgprs, context.required_sgpr_count);
  const uint32_t base = (current + 1u) & ~1u;
  if (base > 126)
    return std::nullopt;

  const uint32_t max_descriptor_sgprs = max_descriptor_sgpr_allocation_for_long_branch(arch);
  if (max_descriptor_sgprs != 0 && base + 2 > max_descriptor_sgprs)
    return std::nullopt;
  return static_cast<uint16_t>(base);
}

[[nodiscard]] std::optional<VirtualLdsBaseSgprReservation>
reserve_virtual_lds_base_sgpr_pair(TranslationContext &context, KernelBlockScope blocks,
                                   rj_code_arch_t arch) {
  if (!is_cdna_arch(arch))
    return std::nullopt;

  // Virtual LDS flat/global operations need an ordinary 64-bit SGPR base, and
  // the entry prologue needs one scalar scratch register to compute the
  // per-workgroup byte offset. The descriptor can encode a 112-SGPR allocation
  // on CDNA because special registers such as VCC live above the ordinary
  // s0..s101 range, but generated DBT code must not allocate those special
  // registers as scratch. Prefer fresh ordinary SGPRs; when only high guest
  // ordinary registers are available, borrow them at entry before guest code has
  // defined them, and preserve the base pair around each lowered LDS use.
  const uint32_t current = std::max(context.num_sgprs, context.required_sgpr_count);
  const uint32_t base = (current + 1u) & ~1u;
  if (base + 4 <= kCdnaOrdinarySgprLimit) {
    context.require_sgprs(base + 4);
    return VirtualLdsBaseSgprReservation{.base = static_cast<uint16_t>(base),
                                         .prologue_temp = static_cast<uint16_t>(base + 2)};
  }

  const uint32_t allocated_ordinary = std::min<uint32_t>(context.num_sgprs, kCdnaOrdinarySgprLimit);
  if (allocated_ordinary < 2)
    return std::nullopt;
  (void)blocks;

  const uint32_t borrowed_temp = (allocated_ordinary - 2u) & ~1u;
  if (base + 2 <= kCdnaOrdinarySgprLimit) {
    context.require_sgprs(base + 2);
    return VirtualLdsBaseSgprReservation{.base = static_cast<uint16_t>(base),
                                         .prologue_temp = static_cast<uint16_t>(borrowed_temp)};
  }

  if (allocated_ordinary < 4)
    return std::nullopt;

  // The descriptor is already full, so do not permanently reuse an existing
  // pair based on static analysis. Borrow the last descriptor-backed ordinary
  // pair only inside each lowered LDS memory sequence, saving and restoring the
  // original scalar value through VGPR temps. A neighboring high SGPR handles
  // entry-only offset math before guest scalar values become meaningful.
  const uint32_t spill_base = (allocated_ordinary - 2u) & ~1u;
  const uint32_t temp_base = (spill_base >= 2) ? ((spill_base - 2u) & ~1u) : 0;
  if (temp_base == spill_base)
    return std::nullopt;
  return VirtualLdsBaseSgprReservation{.base = static_cast<uint16_t>(spill_base),
                                       .prologue_temp = static_cast<uint16_t>(temp_base),
                                       .spill_per_use = true};
}

[[nodiscard]] bool append_virtual_lds_entry_prologue(KdTranslation &translation) {
  if (!translation.needs_lds_overflow_buf)
    return true;
  const uint16_t pointer_base_sgpr = translation.lds_overflow_pointer_in_dispatch_packet
                                         ? translation.dispatch_ptr_sgpr
                                         : translation.kernarg_segment_ptr_sgpr;
  const bool has_pointer_base = translation.lds_overflow_pointer_in_dispatch_packet
                                    ? translation.has_dispatch_ptr
                                    : translation.has_kernarg_segment_ptr;
  if (!has_pointer_base)
    return false;
  if ((translation.lds_overflow_base_sgpr % 2) != 0 || (pointer_base_sgpr % 2) != 0)
    return false;
  if ((translation.lds_overflow_prologue_temp_sgpr % 2) != 0)
    return false;
  if (translation.lds_overflow_base_sgpr + 1 >= kCdnaOrdinarySgprLimit ||
      translation.lds_overflow_prologue_temp_sgpr + 1 >= kCdnaOrdinarySgprLimit ||
      pointer_base_sgpr > 126)
    return false;
  if (translation.lds_overflow_kernarg_pointer_offset > kCdnaSmemImmediateByteOffsetMax)
    return false;
  if (!translation.lds_overflow_pointer_in_dispatch_packet &&
      translation.lds_overflow_kernarg_pointer_offset >
          kCdnaSmemImmediateByteOffsetMax - kVirtualLdsRuntimeStateBytes)
    return false;

  auto valid_workgroup_sgpr = [](int16_t sgpr) { return sgpr < 0 || sgpr <= 126; };
  if (!valid_workgroup_sgpr(translation.workgroup_id_sgpr_x) ||
      !valid_workgroup_sgpr(translation.workgroup_id_sgpr_y) ||
      !valid_workgroup_sgpr(translation.workgroup_id_sgpr_z)) {
    return false;
  }

  const auto base = static_cast<uint8_t>(translation.lds_overflow_base_sgpr);
  const auto temp = static_cast<uint8_t>(translation.lds_overflow_prologue_temp_sgpr);
  const auto product = static_cast<uint8_t>(temp + 1);
  uint8_t state_sbase = static_cast<uint8_t>(pointer_base_sgpr);
  uint32_t state_offset = translation.lds_overflow_kernarg_pointer_offset;

  auto append_smem_load_dword = [&](uint8_t dst, uint8_t sbase, uint32_t offset) {
    auto [load0, load1] = Cdna3MemoryInstructionBuilder::smem_load_dword(dst, sbase, offset);
    translation.prologue_words.push_back(load0);
    translation.prologue_words.push_back(load1);
    translation.prologue_words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntAll0));
  };
  auto append_smem_load_dwordx2 = [&](uint8_t dst, uint8_t sbase, uint32_t offset) {
    auto [load0, load1] = Cdna3MemoryInstructionBuilder::smem_load_dwordx2(dst, sbase, offset);
    translation.prologue_words.push_back(load0);
    translation.prologue_words.push_back(load1);
    translation.prologue_words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntAll0));
  };
  auto append_stride_term = [&](int16_t workgroup_id_sgpr, uint32_t stride_offset) {
    if (workgroup_id_sgpr < 0)
      return;
    append_smem_load_dword(product, state_sbase, state_offset + stride_offset);
    translation.prologue_words.push_back(
        build_cdna3_s_mul_i32(product, static_cast<uint16_t>(workgroup_id_sgpr), product));
    translation.prologue_words.push_back(build_cdna3_s_add_u32(temp, temp, product));
  };

  if (translation.lds_overflow_pointer_in_dispatch_packet) {
    // Zero-kernarg assembly kernels cannot safely have their kernarg pointer
    // redirected. The hook instead stores a pointer to a GPU-visible rocjitsu
    // state block in the dispatch packet; load that state pointer first, then
    // use it as the SMEM base for the real backing pointer and stride fields.
    append_smem_load_dwordx2(base, state_sbase, translation.lds_overflow_kernarg_pointer_offset);
    state_sbase = base;
    state_offset = 0;
  }

  // The hook allocates one backing slice per workgroup. Convert the
  // descriptor-selected workgroup ids into a byte offset using dispatch-time
  // stride fields:
  //   offset = wg_x * stride_x + wg_y * stride_y + wg_z * stride_z.
  translation.prologue_words.push_back(
      build_s_mov_b32(temp, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA3));
  append_stride_term(translation.workgroup_id_sgpr_x, kVirtualLdsStateStrideXOffset);
  append_stride_term(translation.workgroup_id_sgpr_y, kVirtualLdsStateStrideYOffset);
  append_stride_term(translation.workgroup_id_sgpr_z, kVirtualLdsStateStrideZOffset);

  append_smem_load_dwordx2(base, state_sbase, state_offset + kVirtualLdsStateBackingBaseOffset);
  translation.prologue_words.push_back(build_cdna3_s_add_u32(base, base, temp));
  translation.prologue_words.push_back(build_cdna3_s_addc_u32(static_cast<uint8_t>(base + 1),
                                                              static_cast<uint8_t>(base + 1),
                                                              scalar_positive_inline_u32(0)));
  if (translation.lds_overflow_base_sgpr_spill_per_use) {
    if (!translation.lds_overflow_base_pointer_spilled)
      return false;
    if (translation.target_vgpr_count < 2 || translation.target_vgpr_count > 256)
      return false;

    // Spill-per-use mode borrows a guest SGPR pair for each lowered LDS
    // operation. The kernarg/dispatch pointer SGPR pair is not preserved by the
    // guest body, so consume it at the descriptor entry and save the backing
    // pointer in private scratch before any original instruction can clobber
    // the pointer base. Descriptor recomputation after body lowering may raise
    // target_vgpr_count; keep the originally chosen temps stable so the
    // recomputed descriptor reproduces the already-emitted prologue exactly.
    if (!translation.lds_overflow_entry_temp_vgprs_valid) {
      translation.lds_overflow_entry_temp_vgpr_lo =
          static_cast<uint8_t>(translation.target_vgpr_count - 2);
      translation.lds_overflow_entry_temp_vgpr_hi =
          static_cast<uint8_t>(translation.target_vgpr_count - 1);
      translation.lds_overflow_entry_temp_vgprs_valid = true;
    }
    if (translation.lds_overflow_entry_temp_vgpr_hi >= translation.target_vgpr_count)
      return false;
    const uint8_t temp_lo = translation.lds_overflow_entry_temp_vgpr_lo;
    const uint8_t temp_hi = translation.lds_overflow_entry_temp_vgpr_hi;
    translation.prologue_words.push_back(
        build_cdna3_v_mov_b32(temp_lo, static_cast<uint8_t>(translation.lds_overflow_base_sgpr)));
    translation.prologue_words.push_back(build_cdna3_v_mov_b32(
        temp_hi, static_cast<uint8_t>(translation.lds_overflow_base_sgpr + 1)));
    emit_cdna3_scratch_store_b32(translation.prologue_words, temp_lo,
                                 translation.lds_overflow_base_pointer_spill_offset);
    emit_cdna3_scratch_store_b32(translation.prologue_words, temp_hi,
                                 translation.lds_overflow_base_pointer_spill_offset +
                                     sizeof(uint32_t));
    translation.prologue_words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntAll0));
    return true;
  }

  return true;
}

struct VirtualLdsDsOp {
  bool is_load = false;
  uint8_t flat_op = 0;
  uint8_t vgpr_count = 1;
  uint32_t two_addr_stride_bytes = 0;
  uint8_t read2_dst_delta = 0;
};

[[nodiscard]] Cdna3MemoryInstructionBuilder::FlatGlobalOperands
make_virtual_lds_flat_global_operands(uint16_t signed_offset13, uint8_t addr, uint8_t saddr) {
  Cdna3MemoryInstructionBuilder::FlatGlobalOperands operands{};
  operands.signed_offset13 = signed_offset13;
  operands.addr = addr;
  operands.saddr = saddr;
  // Virtual LDS uses global memory as a workgroup-local backing store. On
  // GFX940-class FLAT/GLOBAL instructions, SC[1:0] = 1 encodes group scope,
  // which is the closest memory scope to native LDS producer/consumer traffic.
  operands.sc0 = true;
  operands.sc1 = false;
  return operands;
}

[[nodiscard]] std::optional<VirtualLdsDsOp> virtual_lds_ds_op(uint16_t opcode) {
  switch (opcode) {
  case 13: // ds_write_b32
    return VirtualLdsDsOp{.is_load = false, .flat_op = kCdna3FlatOpStoreDword};
  case 14: // ds_write2_b32
    return VirtualLdsDsOp{.is_load = false,
                          .flat_op = kCdna3FlatOpStoreDword,
                          .vgpr_count = 1,
                          .two_addr_stride_bytes = 4};
  case 15: // ds_write2st64_b32
    return VirtualLdsDsOp{.is_load = false,
                          .flat_op = kCdna3FlatOpStoreDword,
                          .vgpr_count = 1,
                          .two_addr_stride_bytes = 256};
  case 30: // ds_write_b8
    return VirtualLdsDsOp{.is_load = false, .flat_op = kCdna3FlatOpStoreByte};
  case 31: // ds_write_b16
    return VirtualLdsDsOp{.is_load = false, .flat_op = kCdna3FlatOpStoreShort};
  case 54: // ds_read_b32
    return VirtualLdsDsOp{.is_load = true, .flat_op = kCdna3FlatOpLoadDword};
  case 55: // ds_read2_b32
    return VirtualLdsDsOp{.is_load = true,
                          .flat_op = kCdna3FlatOpLoadDword,
                          .vgpr_count = 1,
                          .two_addr_stride_bytes = 4,
                          .read2_dst_delta = 1};
  case 56: // ds_read2st64_b32
    return VirtualLdsDsOp{.is_load = true,
                          .flat_op = kCdna3FlatOpLoadDword,
                          .vgpr_count = 1,
                          .two_addr_stride_bytes = 256,
                          .read2_dst_delta = 1};
  case 57: // ds_read_i8
    return VirtualLdsDsOp{.is_load = true, .flat_op = kCdna3FlatOpLoadSbyte};
  case 58: // ds_read_u8
    return VirtualLdsDsOp{.is_load = true, .flat_op = kCdna3FlatOpLoadUbyte};
  case 60: // ds_read_u16
    return VirtualLdsDsOp{.is_load = true, .flat_op = kCdna3FlatOpLoadUshort};
  case 77: // ds_write_b64
    return VirtualLdsDsOp{.is_load = false, .flat_op = kCdna3FlatOpStoreDword + 1, .vgpr_count = 2};
  case 78: // ds_write2_b64
    return VirtualLdsDsOp{.is_load = false,
                          .flat_op = kCdna3FlatOpStoreDword + 1,
                          .vgpr_count = 2,
                          .two_addr_stride_bytes = 8};
  case 79: // ds_write2st64_b64
    return VirtualLdsDsOp{.is_load = false,
                          .flat_op = kCdna3FlatOpStoreDword + 1,
                          .vgpr_count = 2,
                          .two_addr_stride_bytes = 512};
  case 85: // ds_write_b16_d16_hi
    return VirtualLdsDsOp{.is_load = false, .flat_op = kCdna3FlatOpStoreShortD16Hi};
  case 90: // ds_read_u16_d16
    return VirtualLdsDsOp{.is_load = true, .flat_op = kCdna3FlatOpLoadShortD16};
  case 91: // ds_read_u16_d16_hi
    return VirtualLdsDsOp{.is_load = true, .flat_op = kCdna3FlatOpLoadShortD16Hi};
  case 118: // ds_read_b64
    return VirtualLdsDsOp{.is_load = true, .flat_op = kCdna3FlatOpLoadDword + 1, .vgpr_count = 2};
  case 119: // ds_read2_b64
    return VirtualLdsDsOp{.is_load = true,
                          .flat_op = kCdna3FlatOpLoadDword + 1,
                          .vgpr_count = 2,
                          .two_addr_stride_bytes = 8,
                          .read2_dst_delta = 2};
  case 120: // ds_read2st64_b64
    return VirtualLdsDsOp{.is_load = true,
                          .flat_op = kCdna3FlatOpLoadDword + 1,
                          .vgpr_count = 2,
                          // ST64 offsets are scaled by 64 elements, not by a
                          // fixed byte count. B64 therefore uses 8 * 64 bytes,
                          // matching the write form and AMD's DS pseudocode.
                          .two_addr_stride_bytes = 512,
                          .read2_dst_delta = 2};
  case 222: // ds_write_b96
    return VirtualLdsDsOp{.is_load = false, .flat_op = kCdna3FlatOpStoreDword + 2, .vgpr_count = 3};
  case 223: // ds_write_b128
    return VirtualLdsDsOp{.is_load = false, .flat_op = kCdna3FlatOpStoreDword + 3, .vgpr_count = 4};
  case 254: // ds_read_b96
    return VirtualLdsDsOp{.is_load = true, .flat_op = kCdna3FlatOpLoadDword + 2, .vgpr_count = 3};
  case 255: // ds_read_b128
    return VirtualLdsDsOp{.is_load = true, .flat_op = kCdna3FlatOpLoadDword + 3, .vgpr_count = 4};
  default:
    return std::nullopt;
  }
}

[[nodiscard]] bool is_cdna4_ds_encoding(const Instruction &inst) {
  return (inst.encoding_id() & kCdna4EncodingFamilyMask) == kCdna4DsEncodingFamily;
}

[[nodiscard]] bool source_instruction_uses_virtualizable_lds(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic.find("_lds") != std::string_view::npos)
    return true;
  if (mnemonic == "ds_read_b64_tr_b16")
    return true;
  return is_cdna4_ds_encoding(inst) && virtual_lds_ds_op(inst.opcode()).has_value();
}

[[nodiscard]] ExpandResult lower_virtual_lds_ds_instruction(const Instruction &inst,
                                                            TranslationContext &context) {
  if (!context.virtualize_lds || !is_cdna4_ds_encoding(inst))
    return ExpandResult::not_handled();

  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || static_cast<size_t>(inst.size()) < sizeof(cdna4::DsMachineInst)) {
    return ExpandResult::failed(std::string(inst.mnemonic()) + ": missing DS source encoding",
                                {"Decode the source DS instruction before virtual LDS lowering."});
  }

  cdna4::DsMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const auto op = virtual_lds_ds_op(static_cast<uint16_t>(src.op));
  if (!op) {
    if ((inst.flags() & MEMORY_OP) == 0)
      return ExpandResult::not_handled();
    return ExpandResult::failed(
        std::string(inst.mnemonic()) + ": virtual LDS lowering does not support this DS opcode",
        {"Add a virtual-LDS lowering for this DS memory operation before translating this "
         "kernel with hardware LDS set to zero."});
  }
  if (src.gds != 0) {
    return ExpandResult::failed(std::string(inst.mnemonic()) +
                                    ": virtual LDS lowering does not support GDS",
                                {"Keep virtual LDS lowering limited to ordinary LDS accesses."});
  }
  if (src.acc != 0) {
    return ExpandResult::failed(
        std::string(inst.mnemonic()) + ": virtual LDS lowering does not support ACC operands",
        {"Add AccVGPR-aware virtual LDS load/store lowering before enabling this form."});
  }
  if (context.virtual_lds_base_sgpr > 126 || (context.virtual_lds_base_sgpr % 2) != 0) {
    return ExpandResult::failed(
        std::string(inst.mnemonic()) + ": virtual LDS backing-buffer SGPR pair is not encodable",
        {"Reserve an even SGPR pair that CDNA3 flat/global instructions can encode as saddr."});
  }

  std::vector<uint32_t> words;
  if (op->two_addr_stride_bytes != 0 && op->is_load) {
    const uint32_t byte_offset0 = static_cast<uint32_t>(src.offset0) * op->two_addr_stride_bytes;
    const uint32_t byte_offset1 = static_cast<uint32_t>(src.offset1) * op->two_addr_stride_bytes;
    const uint16_t total_dst_vgprs = static_cast<uint16_t>(op->read2_dst_delta + op->vgpr_count);
    const bool first_load_clobbers_addr = virtual_lds_vgpr_ranges_overlap(
        static_cast<uint16_t>(src.vdst), op->vgpr_count, static_cast<uint16_t>(src.addr), 1);
    const bool needs_materialized_offset =
        byte_offset0 > kFlatGlobalPositiveImm13Max || byte_offset1 > kFlatGlobalPositiveImm13Max;

    std::vector<VirtualLdsVgprRange> forbidden = {
        {.base = static_cast<uint16_t>(src.addr), .count = 1},
        {.base = static_cast<uint16_t>(src.vdst), .count = total_dst_vgprs},
    };
    std::optional<VirtualLdsAddressTemp> base_temp;
    if (first_load_clobbers_addr) {
      base_temp = choose_virtual_lds_address_temp(context, forbidden);
      if (!base_temp) {
        return ExpandResult::failed(
            std::string(inst.mnemonic()) +
                ": virtual LDS read2 lowering cannot preserve an overlapping address VGPR",
            {"Add a more general spill path for read2 forms whose first destination clobbers the "
             "address operand."});
      }
      forbidden.push_back({.base = base_temp->reg, .count = 1});
    }

    std::optional<VirtualLdsAddressTemp> offset_temp;
    if (needs_materialized_offset) {
      offset_temp = choose_virtual_lds_address_temp(context, forbidden);
      if (!offset_temp) {
        return ExpandResult::failed(
            std::string(inst.mnemonic()) +
                ": virtual LDS read2 lowering cannot find a VGPR for offset materialization",
            {"Add a more general spill path for read2 forms whose operands cover every ordinary "
             "VGPR."});
      }
      forbidden.push_back({.base = offset_temp->reg, .count = 1});
    }

    std::optional<std::array<VirtualLdsAddressTemp, 2>> base_spill_temps;
    if (context.virtual_lds_base_sgpr_spill_per_use) {
      base_spill_temps = choose_virtual_lds_base_spill_temps(context, forbidden);
      if (!base_spill_temps) {
        return ExpandResult::failed(
            std::string(inst.mnemonic()) +
                ": virtual LDS SGPR spill lowering cannot find VGPR save temps",
            {"Add a more general spill path for kernels whose DS operands cover every ordinary "
             "VGPR."});
      }
    }

    const uint32_t base_sgpr_save_offset = assign_virtual_lds_spill_offsets(
        context,
        {base_temp ? &*base_temp : nullptr, offset_temp ? &*offset_temp : nullptr,
         base_spill_temps ? &(*base_spill_temps)[0] : nullptr,
         base_spill_temps ? &(*base_spill_temps)[1] : nullptr},
        base_spill_temps && context.virtual_lds_base_pointer_spilled ? 2u : 0u);
    emit_virtual_lds_temp_spill_stores(
        words, {base_temp ? &*base_temp : nullptr, offset_temp ? &*offset_temp : nullptr,
                base_spill_temps ? &(*base_spill_temps)[0] : nullptr,
                base_spill_temps ? &(*base_spill_temps)[1] : nullptr});
    if (base_spill_temps)
      emit_virtual_lds_base_spill_setup(words, context, *base_spill_temps, base_sgpr_save_offset);

    // Native ds_read2 issues two LDS reads from the same original address VGPR.
    // If the first destination aliases that address VGPR, keep a private copy so
    // the second flat/global load still observes the pre-instruction address.
    const uint8_t preserved_base = base_temp ? base_temp->reg : static_cast<uint8_t>(src.addr);
    if (base_temp)
      emit_cdna3_v_add_u32_literal(words, preserved_base, static_cast<uint8_t>(src.addr), 0);

    auto emit_read2_load = [&](uint8_t vdst, uint32_t byte_offset) {
      uint8_t addr = preserved_base;
      uint16_t flat_offset = static_cast<uint16_t>(byte_offset);
      if (byte_offset > kFlatGlobalPositiveImm13Max) {
        addr = offset_temp->reg;
        flat_offset = 0;
        emit_cdna3_v_add_u32_literal(words, addr, preserved_base, byte_offset);
      }

      const auto operands = make_virtual_lds_flat_global_operands(
          flat_offset, addr, static_cast<uint8_t>(context.virtual_lds_base_sgpr));
      auto [w0, w1] = Cdna3MemoryInstructionBuilder::flat_global_load(operands, op->flat_op, vdst);
      words.push_back(w0);
      words.push_back(w1);
    };

    emit_read2_load(static_cast<uint8_t>(src.vdst), byte_offset0);
    emit_read2_load(static_cast<uint8_t>(src.vdst + op->read2_dst_delta), byte_offset1);
    // Native DS participates in lgkmcnt. Complete the replacement VMEM before
    // exposing control to instructions that were scheduled around LDS waits.
    words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntAll0));
    if (base_spill_temps)
      emit_virtual_lds_base_spill_restore(words, context, *base_spill_temps, base_sgpr_save_offset);
    emit_virtual_lds_temp_spill_loads(words, {base_spill_temps ? &(*base_spill_temps)[1] : nullptr,
                                              base_spill_temps ? &(*base_spill_temps)[0] : nullptr,
                                              offset_temp ? &*offset_temp : nullptr,
                                              base_temp ? &*base_temp : nullptr});
    if (!guard_virtual_lds_execz(words, context)) {
      return ExpandResult::failed(
          std::string(inst.mnemonic()) + ": virtual LDS SGPR spill guard branch is out of range",
          {"Reduce the per-instruction virtual-LDS spill sequence or add a long guard branch."});
    }
    return ExpandResult::success(std::move(words));
  }

  if (op->two_addr_stride_bytes != 0) {
    const uint32_t byte_offset0 = static_cast<uint32_t>(src.offset0) * op->two_addr_stride_bytes;
    const uint32_t byte_offset1 = static_cast<uint32_t>(src.offset1) * op->two_addr_stride_bytes;
    const bool needs_materialized_offset =
        byte_offset0 > kFlatGlobalPositiveImm13Max || byte_offset1 > kFlatGlobalPositiveImm13Max;

    std::optional<VirtualLdsAddressTemp> offset_temp;
    std::vector<VirtualLdsVgprRange> forbidden = {
        {.base = static_cast<uint16_t>(src.addr), .count = 1},
        {.base = static_cast<uint16_t>(src.data0), .count = op->vgpr_count},
        {.base = static_cast<uint16_t>(src.data1), .count = op->vgpr_count},
    };
    if (needs_materialized_offset) {
      // Store forms do not clobber their VGPR sources, but the temporary used
      // to materialize a large byte offset must not overlap the address or
      // either data window before the flat/global store consumes them.
      offset_temp = choose_virtual_lds_address_temp(context, forbidden);
      if (!offset_temp) {
        return ExpandResult::failed(
            std::string(inst.mnemonic()) +
                ": virtual LDS write2 lowering cannot find a VGPR for offset materialization",
            {"Add a more general spill path for write2 forms whose operands cover every ordinary "
             "VGPR."});
      }
      forbidden.push_back({.base = offset_temp->reg, .count = 1});
    }

    std::optional<std::array<VirtualLdsAddressTemp, 2>> base_spill_temps;
    if (context.virtual_lds_base_sgpr_spill_per_use) {
      base_spill_temps = choose_virtual_lds_base_spill_temps(context, forbidden);
      if (!base_spill_temps) {
        return ExpandResult::failed(
            std::string(inst.mnemonic()) +
                ": virtual LDS SGPR spill lowering cannot find VGPR save temps",
            {"Add a more general spill path for kernels whose DS operands cover every ordinary "
             "VGPR."});
      }
    }

    const uint32_t base_sgpr_save_offset = assign_virtual_lds_spill_offsets(
        context,
        {offset_temp ? &*offset_temp : nullptr,
         base_spill_temps ? &(*base_spill_temps)[0] : nullptr,
         base_spill_temps ? &(*base_spill_temps)[1] : nullptr},
        base_spill_temps && context.virtual_lds_base_pointer_spilled ? 2u : 0u);
    emit_virtual_lds_temp_spill_stores(words,
                                       {offset_temp ? &*offset_temp : nullptr,
                                        base_spill_temps ? &(*base_spill_temps)[0] : nullptr,
                                        base_spill_temps ? &(*base_spill_temps)[1] : nullptr});
    if (base_spill_temps)
      emit_virtual_lds_base_spill_setup(words, context, *base_spill_temps, base_sgpr_save_offset);

    auto emit_write2_store = [&](uint8_t data, uint32_t byte_offset) {
      uint8_t addr = static_cast<uint8_t>(src.addr);
      uint16_t flat_offset = static_cast<uint16_t>(byte_offset);
      if (byte_offset > kFlatGlobalPositiveImm13Max) {
        addr = offset_temp->reg;
        flat_offset = 0;
        emit_cdna3_v_add_u32_literal(words, addr, static_cast<uint8_t>(src.addr), byte_offset);
      }

      const auto operands = make_virtual_lds_flat_global_operands(
          flat_offset, addr, static_cast<uint8_t>(context.virtual_lds_base_sgpr));
      auto [w0, w1] = Cdna3MemoryInstructionBuilder::flat_global_store(operands, op->flat_op, data);
      words.push_back(w0);
      words.push_back(w1);
    };

    emit_write2_store(static_cast<uint8_t>(src.data0), byte_offset0);
    emit_write2_store(static_cast<uint8_t>(src.data1), byte_offset1);
    words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntAll0));
    if (base_spill_temps)
      emit_virtual_lds_base_spill_restore(words, context, *base_spill_temps, base_sgpr_save_offset);
    emit_virtual_lds_temp_spill_loads(words, {base_spill_temps ? &(*base_spill_temps)[1] : nullptr,
                                              base_spill_temps ? &(*base_spill_temps)[0] : nullptr,
                                              offset_temp ? &*offset_temp : nullptr});
    if (!guard_virtual_lds_execz(words, context)) {
      return ExpandResult::failed(
          std::string(inst.mnemonic()) + ": virtual LDS SGPR spill guard branch is out of range",
          {"Reduce the per-instruction virtual-LDS spill sequence or add a long guard branch."});
    }
    return ExpandResult::success(std::move(words));
  }

  const uint32_t ds_offset = (static_cast<uint32_t>(src.offset1) << 8) | src.offset0;

  uint8_t addr = static_cast<uint8_t>(src.addr);
  uint16_t flat_offset = static_cast<uint16_t>(ds_offset);
  std::optional<VirtualLdsAddressTemp> address_temp;
  std::vector<VirtualLdsVgprRange> forbidden = {
      {.base = static_cast<uint16_t>(src.addr), .count = 1},
      {.base = op->is_load ? static_cast<uint16_t>(src.vdst) : static_cast<uint16_t>(src.data0),
       .count = op->vgpr_count},
  };
  if (ds_offset > kFlatGlobalPositiveImm13Max) {
    address_temp = choose_virtual_lds_address_temp(context, forbidden);
    if (!address_temp) {
      return ExpandResult::failed(
          std::string(inst.mnemonic()) +
              ": virtual LDS lowering cannot find a VGPR for large DS offset materialization",
          {"Add a more general spill path for kernels whose DS operands cover every ordinary "
           "VGPR."});
    }
    forbidden.push_back({.base = address_temp->reg, .count = 1});
  }

  std::optional<std::array<VirtualLdsAddressTemp, 2>> base_spill_temps;
  if (context.virtual_lds_base_sgpr_spill_per_use) {
    base_spill_temps = choose_virtual_lds_base_spill_temps(context, forbidden);
    if (!base_spill_temps) {
      return ExpandResult::failed(
          std::string(inst.mnemonic()) +
              ": virtual LDS SGPR spill lowering cannot find VGPR save temps",
          {"Add a more general spill path for kernels whose DS operands cover every ordinary "
           "VGPR."});
    }
  }

  const uint32_t base_sgpr_save_offset = assign_virtual_lds_spill_offsets(
      context,
      {address_temp ? &*address_temp : nullptr,
       base_spill_temps ? &(*base_spill_temps)[0] : nullptr,
       base_spill_temps ? &(*base_spill_temps)[1] : nullptr},
      base_spill_temps && context.virtual_lds_base_pointer_spilled ? 2u : 0u);
  emit_virtual_lds_temp_spill_stores(words, {address_temp ? &*address_temp : nullptr,
                                             base_spill_temps ? &(*base_spill_temps)[0] : nullptr,
                                             base_spill_temps ? &(*base_spill_temps)[1] : nullptr});
  if (base_spill_temps)
    emit_virtual_lds_base_spill_setup(words, context, *base_spill_temps, base_sgpr_save_offset);

  if (address_temp) {
    emit_cdna3_v_add_u32_literal(words, address_temp->reg, static_cast<uint8_t>(src.addr),
                                 ds_offset);
    addr = address_temp->reg;
    flat_offset = 0;
  }

  const auto operands = make_virtual_lds_flat_global_operands(
      flat_offset, addr, static_cast<uint8_t>(context.virtual_lds_base_sgpr));

  if (op->is_load) {
    auto [w0, w1] = Cdna3MemoryInstructionBuilder::flat_global_load(operands, op->flat_op,
                                                                    static_cast<uint8_t>(src.vdst));
    words.push_back(w0);
    words.push_back(w1);
  } else {
    auto [w0, w1] = Cdna3MemoryInstructionBuilder::flat_global_store(
        operands, op->flat_op, static_cast<uint8_t>(src.data0));
    words.push_back(w0);
    words.push_back(w1);
  }
  // Native DS participates in lgkmcnt. A flat/global replacement participates
  // in VMEM counters instead, so a conservative all-counter wait preserves the
  // local completion point for code scheduled around LDS memory operations.
  words.push_back(pack_sopp(kCdna3SoppOpWaitcnt, kCdnaWaitcntAll0));
  if (base_spill_temps)
    emit_virtual_lds_base_spill_restore(words, context, *base_spill_temps, base_sgpr_save_offset);
  emit_virtual_lds_temp_spill_loads(words, {base_spill_temps ? &(*base_spill_temps)[1] : nullptr,
                                            base_spill_temps ? &(*base_spill_temps)[0] : nullptr,
                                            address_temp ? &*address_temp : nullptr});
  if (!guard_virtual_lds_execz(words, context)) {
    return ExpandResult::failed(
        std::string(inst.mnemonic()) + ": virtual LDS SGPR spill guard branch is out of range",
        {"Reduce the per-instruction virtual-LDS spill sequence or add a long guard branch."});
  }
  return ExpandResult::success(std::move(words));
}

[[nodiscard]] std::vector<uint64_t> kernel_entry_offsets(std::span<const KdTranslation> kernels) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size());
  for (const KdTranslation &kernel : kernels)
    offsets.push_back(kernel.entry_text_offset);

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

[[nodiscard]] std::vector<uint64_t>
kernel_hardware_entry_offsets(std::span<const KdTranslation> kernels) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size() * 2);
  for (const KdTranslation &kernel : kernels) {
    offsets.push_back(kernel.entry_text_offset);
    if (kernel.has_kernarg_preload)
      offsets.push_back(kernel.kernarg_preload_entry_text_offset);
  }

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

[[nodiscard]] std::vector<uint64_t> kernel_block_leaders(std::span<const KdTranslation> kernels,
                                                         std::span<const uint8_t> text) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size() * 2);
  for (const KdTranslation &kernel : kernels) {
    offsets.push_back(kernel.entry_text_offset);
    // AMDHSA kernarg preloading is descriptor-controlled. When
    // kernarg_preload_spec_length is non-zero, compatible CP firmware starts at
    // KERNEL_CODE_ENTRY_BYTE_OFFSET + 256. That address is a real hardware entry,
    // not merely padding, so split a block there and seed reachability from it.
    if (kernel.has_kernarg_preload && kernel.kernarg_preload_entry_text_offset < text.size())
      offsets.push_back(kernel.kernarg_preload_entry_text_offset);
  }

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

struct KernelTranslationScope {
  KdTranslation *translation = nullptr;
  BasicBlock *entry = nullptr;
  std::vector<BasicBlock *> blocks;
};

[[nodiscard]] uint64_t kernel_scope_key(const KdTranslation &kernel) {
  assert(kernel.entry_text_offset <= (std::numeric_limits<uint64_t>::max() >> 1) &&
         "kernel entry offset is too large to pack with variant bit");
  return (kernel.entry_text_offset << 1) | (kernel.needs_lds_overflow_buf ? 1u : 0u);
}

[[nodiscard]] bool same_kernel_scope_variant(const KdTranslation &lhs, const KdTranslation &rhs) {
  return lhs.entry_text_offset == rhs.entry_text_offset &&
         lhs.needs_lds_overflow_buf == rhs.needs_lds_overflow_buf;
}

[[nodiscard]] size_t kernel_translation_scope_count(std::span<const KdTranslation> kernels) {
  std::unordered_set<uint64_t> keys;
  for (const KdTranslation &kernel : kernels)
    keys.insert(kernel_scope_key(kernel));
  return keys.size();
}

[[nodiscard]] bool scope_uses_virtualizable_lds(const KernelTranslationScope &scope) {
  if (scope.translation == nullptr)
    return false;
  if (scope.translation->target_lds_size != 0)
    return true;

  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (source_instruction_uses_virtualizable_lds(inst))
        return true;
    }
  }
  return false;
}

/// @brief Sorted index from source .text byte offsets to decoded blocks.
///
/// @details DBT relocation repeatedly maps descriptor entries, branch targets,
/// and recovered indirect targets back to the BasicBlock that owns a source
/// offset. Keeping this compact sorted index avoids rebuilding that lookup while
/// preserving BasicBlock ownership in the vector returned by BasicBlock::build().
using BlockOffsetIndex = std::vector<std::pair<uint64_t, BasicBlock *>>;
using BlockPositionIndex = std::unordered_map<const BasicBlock *, size_t>;

[[nodiscard]] BlockOffsetIndex
build_block_offset_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  BlockOffsetIndex index;
  index.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block != nullptr)
      index.emplace_back(block->start_offset(), block.get());
  }
  std::ranges::sort(index, {}, &std::pair<uint64_t, BasicBlock *>::first);
  return index;
}

[[nodiscard]] BlockPositionIndex
build_block_position_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  BlockPositionIndex index;
  index.reserve(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr)
      index.emplace(blocks[i].get(), i);
  }
  return index;
}

[[nodiscard]] BasicBlock *block_for_offset(const BlockOffsetIndex &index, uint64_t offset) {
  auto it = std::ranges::upper_bound(index, offset, std::less<>{},
                                     &std::pair<uint64_t, BasicBlock *>::first);
  if (it == index.begin())
    return nullptr;
  --it;

  BasicBlock *block = it->second;
  if (block == nullptr || offset >= block->end_offset())
    return nullptr;
  return block;
}

[[nodiscard]] std::vector<BasicBlock *>
reachable_kernel_blocks(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                        const BlockOffsetIndex &block_index,
                        const BlockPositionIndex &block_positions, BasicBlock &entry,
                        const std::unordered_set<uint64_t> &kernel_entries,
                        const std::unordered_set<uint64_t> &own_entries) {
  std::vector<uint8_t> reachable(blocks.size(), 0);
  std::vector<size_t> reached_indices;
  std::vector<size_t> stack;
  auto push_block = [&](BasicBlock *block) {
    auto it = block_positions.find(block);
    if (it != block_positions.end())
      stack.push_back(it->second);
  };
  push_block(&entry);
  for (const uint64_t own_entry : own_entries) {
    if (own_entry == entry.start_offset())
      continue;
    if (BasicBlock *extra_entry = block_for_offset(block_index, own_entry);
        extra_entry != nullptr && extra_entry != &entry) {
      push_block(extra_entry);
    }
  }

  while (!stack.empty()) {
    const size_t block_idx = stack.back();
    stack.pop_back();
    if (block_idx >= blocks.size() || reachable[block_idx])
      continue;
    reachable[block_idx] = 1;
    reached_indices.push_back(block_idx);
    BasicBlock *block = blocks[block_idx].get();
    assert(block != nullptr && "reachable walk stack should contain only decoded blocks");

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      if (!own_entries.contains(succ->start_offset()) &&
          kernel_entries.contains(succ->start_offset()))
        continue;
      push_block(succ);
    }
    // Ordinary CFG successors describe control that always follows from the
    // current program counter: fallthroughs, conditional targets, direct branch
    // targets, and recovered non-returning setpc targets. Call edges are tracked
    // separately because a shared callee block can return to different
    // continuations depending on which call site entered it. Reachability for
    // translation still has to include the callee body, but later liveness gets
    // explicit call/return edges rather than treating every possible return as a
    // global CFG successor.
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      BasicBlock *callee = call.callee;
      assert(callee != nullptr && "BasicBlock call edges should always have a callee");
      if (!own_entries.contains(callee->start_offset()) &&
          kernel_entries.contains(callee->start_offset()))
        continue;
      push_block(callee);
    }
  }

  std::ranges::sort(reached_indices);
  std::vector<BasicBlock *> ordered;
  ordered.reserve(reached_indices.size());
  for (size_t block_idx : reached_indices) {
    if (blocks[block_idx])
      ordered.push_back(blocks[block_idx].get());
  }
  return ordered;
}

[[nodiscard]] std::vector<KernelTranslationScope>
kernel_translation_scopes(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                          const BlockOffsetIndex &block_index, std::span<KdTranslation> kernels) {
  std::vector<KernelTranslationScope> scopes;
  const auto entries = kernel_entry_offsets(kernels);
  if (entries.empty())
    return scopes;

  const BlockPositionIndex block_positions = build_block_position_index(blocks);
  const auto hardware_entries = kernel_hardware_entry_offsets(kernels);
  std::unordered_set<uint64_t> entry_set(hardware_entries.begin(), hardware_entries.end());
  std::vector<KdTranslation *> ordered_kernels;
  ordered_kernels.reserve(kernels.size());
  std::unordered_set<uint64_t> seen_scopes;
  for (KdTranslation &kernel : kernels) {
    if (seen_scopes.insert(kernel_scope_key(kernel)).second)
      ordered_kernels.push_back(&kernel);
  }

  std::ranges::sort(ordered_kernels, [](const auto *lhs, const auto *rhs) {
    if (lhs->entry_text_offset != rhs->entry_text_offset)
      return lhs->entry_text_offset < rhs->entry_text_offset;
    return lhs->needs_lds_overflow_buf < rhs->needs_lds_overflow_buf;
  });

  scopes.reserve(ordered_kernels.size());
  for (KdTranslation *kernel : ordered_kernels) {
    BasicBlock *entry = block_for_offset(block_index, kernel->entry_text_offset);
    if (entry == nullptr)
      continue;
    std::unordered_set<uint64_t> own_entries{kernel->entry_text_offset};
    if (kernel->has_kernarg_preload) {
      if (block_for_offset(block_index, kernel->kernarg_preload_entry_text_offset) == nullptr)
        continue;
      own_entries.insert(kernel->kernarg_preload_entry_text_offset);
    }

    scopes.push_back({kernel, entry,
                      reachable_kernel_blocks(blocks, block_index, block_positions, *entry,
                                              entry_set, own_entries)});
  }
  return scopes;
}

/// @brief Return whether an instruction is an `s_setpc_b64` through one SGPR pair.
///
/// @details Return-like scalar control flow is left as an indirect branch in the
/// translated instruction stream, so DBT must validate that the block terminator
/// reads the call edge's saved return SGPR. This helper intentionally checks the
/// raw SOP1 source field instead of broader instruction semantics: only the exact
/// `s_setpc_b64 s[return:return+1]` form is a scoped call return.
[[nodiscard]] bool s_setpc_from_sreg(const Instruction &inst, uint32_t word, uint16_t ssrc0) {
  if (inst.size() != sizeof(uint32_t) || inst.mnemonic() != "s_setpc_b64")
    return false;
  return static_cast<uint16_t>(word & 0xffu) == ssrc0;
}

/// @brief Find return blocks inside one context-sensitive call target.
///
/// @details Call-like scalar control flow is not represented as a normal CFG
/// edge from the callee back to every possible continuation. The same helper
/// block can be entered by multiple kernels or multiple call sites, and the
/// correct continuation is the one selected by the return SGPR written at that
/// call site. This walk therefore stays inside @p allowed_blocks, follows only
/// ordinary successors within the callee body, and reports terminators that
/// return through @p return_sreg. The caller then pairs each return with the
/// specific continuation from the call edge being analyzed.
[[nodiscard]] std::vector<BasicBlock *>
function_return_blocks(BasicBlock &callee, uint16_t return_sreg, std::span<const uint8_t> text,
                       const std::unordered_set<BasicBlock *> &allowed_blocks) {
  std::vector<BasicBlock *> returns;
  std::vector<BasicBlock *> stack{&callee};
  std::unordered_set<BasicBlock *> visited;

  while (!stack.empty()) {
    BasicBlock *block = stack.back();
    stack.pop_back();
    assert(block != nullptr && "return-block walk stack should contain only decoded blocks");
    if (!allowed_blocks.contains(block) || !visited.insert(block).second)
      continue;

    const Instruction *term = block->terminator();
    assert(term != nullptr && "decoded BasicBlock should contain at least one instruction");
    if (s_setpc_from_sreg(*term, text_word_at(text, term->src_loc()), return_sreg)) {
      returns.push_back(block);
      continue;
    }

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      stack.push_back(succ);
    }
  }

  return returns;
}

/// @brief Collect validated return-like terminators for one kernel scope.
///
/// @details Binary translation rejects unresolved indirect branches after CFG
/// construction, but a call-return `s_setpc_b64` is intentionally left as an
/// indirect instruction in the emitted code: its dynamic target is the return PC
/// saved by the matching `s_call_b64` or `s_swappc_b64`. To avoid accepting an
/// arbitrary `s_setpc_b64`, this helper only marks return offsets that are
/// reachable from a `BasicBlock::CallEdge` whose callee and continuation both
/// belong to the current kernel-local scope.
[[nodiscard]] std::unordered_set<uint64_t>
scoped_call_return_offsets(std::span<BasicBlock *const> blocks, std::span<const uint8_t> text) {
  std::unordered_set<BasicBlock *> allowed_blocks;
  allowed_blocks.reserve(blocks.size());
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    allowed_blocks.insert(block);
  }

  std::unordered_set<uint64_t> returns;
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      assert(call.callee != nullptr && "BasicBlock call edges should always have a callee");
      assert(call.continuation != nullptr &&
             "BasicBlock call edges should always have a continuation");
      if (!allowed_blocks.contains(call.callee) || !allowed_blocks.contains(call.continuation))
        continue;

      for (BasicBlock *return_block :
           function_return_blocks(*call.callee, call.return_sreg, text, allowed_blocks)) {
        const Instruction *term = return_block->terminator();
        assert(term != nullptr && "function_return_blocks returns non-empty decoded blocks");
        returns.insert(term->src_loc());
      }
    }
  }
  return returns;
}

/// @brief Materialize context-sensitive call edges for liveness.
///
/// @details `BasicBlock` deliberately separates call edges from ordinary CFG
/// successors. The translator still needs liveness to see the effects of a
/// call: values live into the callee are used by the callee, and values live
/// after the call continuation must be live at each validated return block.
/// This helper converts each scoped call edge into temporary analysis edges
/// `caller -> callee` and `return -> continuation` without mutating the CFG or
/// creating cross-kernel return edges.
[[nodiscard]] std::vector<ScopedCfgEdge>
scoped_call_liveness_edges(std::span<BasicBlock *const> blocks, std::span<const uint8_t> text) {
  std::unordered_set<BasicBlock *> allowed_blocks;
  allowed_blocks.reserve(blocks.size());
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    allowed_blocks.insert(block);
  }

  std::vector<ScopedCfgEdge> edges;
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      assert(call.callee != nullptr && "BasicBlock call edges should always have a callee");
      assert(call.continuation != nullptr &&
             "BasicBlock call edges should always have a continuation");
      if (!allowed_blocks.contains(call.callee) || !allowed_blocks.contains(call.continuation))
        continue;

      edges.push_back({.from = block, .to = call.callee});
      for (BasicBlock *return_block :
           function_return_blocks(*call.callee, call.return_sreg, text, allowed_blocks)) {
        edges.push_back({.from = return_block, .to = call.continuation});
      }
    }
  }

  return edges;
}

} // namespace

BinaryTranslator::~BinaryTranslator() = default;

BinaryTranslator::BinaryTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                   uint32_t target_mach, BinaryTranslatorOptions options)
    : guest_arch_(guest_arch), host_arch_(host_arch),
      target_mach_(target_mach ? target_mach : elf_mach_for_arch(host_arch)), options_(options),
      encoding_translate_(select_encoding_translator(guest_arch, host_arch)),
      legalization_lookup_(select_legalization(guest_arch, host_arch)),
      semantic_translator_(std::make_unique<SemanticTranslator>(guest_arch, host_arch)) {}

void BinaryTranslator::set_trace_callback(TranslationTraceCallback callback) {
  trace_callback_ = std::move(callback);
}

TranslatedCodeObject BinaryTranslator::translate(const AmdGpuCodeObject &obj) {
  TranslatedCodeObject result;
  result.host_arch = host_arch_;
  diagnostics_ = &result.diagnostics;

  CodeObjectPatcher patcher(obj);
  auto leave_unchanged = [&]() {
    diagnostics_ = nullptr;
    const auto *image = reinterpret_cast<const uint8_t *>(obj.image_data());
    result.elf_bytes.assign(image, image + obj.image_size());
    return result;
  };
  auto text = patcher.text_bytes();
  if (text.empty()) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "code object does not expose a non-empty .text section for translation");
    return leave_unchanged();
  }

  // Per-kernel text relocation strategy:
  // 1. Decode kernel descriptors and use their entry offsets as translation roots.
  // 2. Decode .text into basic blocks and recover static indirect-branch targets.
  // 3. Compute each kernel's reachable block set. Ordinary CFG successors are
  //    followed directly; call_edges() are followed only to include the callee
  //    body in the current kernel-local scope.
  // 4. Emit each kernel's reachable blocks into a compact, source-ordered body.
  // 5. Translate instructions with a forward cursor; oversized replacements grow
  //    the current block and recovered indirect transfers reserve patch windows.
  // 6. Patch direct PC-relative branches through the kernel-local placement map.
  // 7. Patch recovered indirect branches/calls either at their reserved transfer
  //    windows or, for multi-target consumers, by rewriting the recovered
  //    source-side PC builders once per distinct builder range. BasicBlock
  //    models setpc-style targets as ordinary CFG successors and validated
  //    swappc calls as call_edges(), so DBT can add return continuations only
  //    inside the kernel scope that owns the call site.
  // 8. Replace the ELF .text payload and redirect descriptors to their new entries.
  auto decoder = Decoder::create(guest_arch_);
  if (!decoder) {
    append_error(result.diagnostics, DiagnosticKind::UnsupportedGuestArch,
                 "unsupported guest_arch: no decoder available");
    return leave_unchanged();
  }

  // Phase 1: descriptor translation gives DBT the source kernel roots and any
  // target descriptor/prologue bytes that must be materialized with the body.
  const bool skip_failed_kernels = options_.skip_failed_kernels;
  KernelDescriptorTranslator descriptor_translator(guest_arch_, host_arch_);
  const bool can_emit_virtual_lds_variants = !options_.force_virtual_lds &&
                                             guest_arch_ == ROCJITSU_CODE_ARCH_CDNA4 &&
                                             host_arch_ == ROCJITSU_CODE_ARCH_CDNA3;
  KernelDescriptorTranslationOptions initial_descriptor_options;
  initial_descriptor_options.virtualize_lds = options_.force_virtual_lds;
  initial_descriptor_options.allow_oversized_lds = can_emit_virtual_lds_variants;
  auto descriptor_translations =
      descriptor_translator.translate_image(patcher.image_bytes(), patcher.text_offset(),
                                            patcher.text_size(), initial_descriptor_options);
  bool descriptors_supported = true;
  for (const auto &translation : descriptor_translations) {
    if (translation.supported || !skip_failed_kernels)
      append_diagnostics(result.diagnostics, translation.diagnostics);
    descriptors_supported &= translation.supported;
  }
  if (!descriptors_supported && !skip_failed_kernels) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor translation requires unsupported resource or ABI "
                 "virtualization; leaving code object unchanged");
    return leave_unchanged();
  }

  if (descriptor_translations.empty()) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptors are required for kernel-level translation");
    return leave_unchanged();
  }

  auto block_leaders = kernel_block_leaders(descriptor_translations, text);

  // Phase 2: build a CFG over .text, including recovered indirect targets as
  // block leaders, then compute one source-reachable block set per descriptor
  // root. These sets are intentionally kernel-local: if two roots reach the same
  // helper block, Phase 3 emits that helper into both relocated bodies so every
  // branch or call target can be resolved through the current kernel's placement
  // map without borrowing another kernel's return continuation.
  auto blocks = BasicBlock::build(obj, *decoder, guest_arch_, block_leaders);
  const BlockOffsetIndex block_index = build_block_offset_index(blocks);
  auto scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations);

  if (can_emit_virtual_lds_variants) {
    std::vector<KdTranslation> virtual_variants;
    for (const KernelTranslationScope &scope : scopes) {
      if (scope.translation == nullptr)
        continue;
      const bool static_lds_exceeds_host =
          scope.translation->target_lds_size > kCdna3MaxHardwareLdsBytes;
      if (!static_lds_exceeds_host &&
          (!scope.translation->has_kernarg_segment_ptr || !scope_uses_virtualizable_lds(scope))) {
        continue;
      }

      KernelDescriptorTranslationOptions virtual_descriptor_options;
      virtual_descriptor_options.virtualize_lds = true;
      auto virtual_translation = descriptor_translator.translate_descriptor(
          patcher.image_bytes(), scope.translation->descriptor_file_offset,
          scope.translation->entry_text_offset, virtual_descriptor_options);
      if (!virtual_translation) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "virtual LDS sidecar descriptor could not be computed; leaving code object "
                     "unchanged",
                     scope.translation->entry_text_offset);
        return leave_unchanged();
      }
      virtual_translation->kernel_name = scope.translation->kernel_name;
      virtual_translation->virtual_lds_variant = true;
      virtual_variants.push_back(std::move(*virtual_translation));
    }
    if (!virtual_variants.empty()) {
      descriptor_translations.insert(descriptor_translations.end(),
                                     std::make_move_iterator(virtual_variants.begin()),
                                     std::make_move_iterator(virtual_variants.end()));
      scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations);
    }
  }

  const size_t expected_scope_count = kernel_translation_scope_count(descriptor_translations);
  if (scopes.size() != expected_scope_count) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor entry offsets are required to map to decoded text blocks");
    return leave_unchanged();
  }

  std::vector<uint8_t> translated_text;
  translated_text.reserve(text.size());
  const bool continue_after_failure = options_.debug_continue_after_failure;

  struct PendingTrace {
    uint64_t source_offset = 0;
    uint32_t source_size = 0;
    std::vector<uint32_t> source_words;
    const InstructionLegalization *legalization = nullptr;
    bool copied_original = false;
    bool semantic_lowering = false;
    bool changed = false;
    uint64_t target_offset = 0;
    std::vector<uint32_t> target_words;
  };

  auto queue_trace = [&](std::vector<PendingTrace> &pending, const Instruction &inst,
                         uint64_t offset, const InstructionLegalization *leg, bool copied_original,
                         bool semantic_lowering, bool changed, uint64_t target_offset,
                         std::vector<uint32_t> target_words) {
    if (!trace_callback_)
      return;
    pending.push_back({.source_offset = offset,
                       .source_size = static_cast<uint32_t>(inst.size()),
                       .source_words = raw_words_for_inst(inst),
                       .legalization = leg,
                       .copied_original = copied_original,
                       .semantic_lowering = semantic_lowering,
                       .changed = changed,
                       .target_offset = target_offset,
                       .target_words = std::move(target_words)});
  };

  auto flush_traces = [&](std::vector<PendingTrace> &pending, uint64_t target_delta) {
    if (!trace_callback_)
      return;
    for (PendingTrace &trace : pending) {
      trace_callback_({.source_offset = trace.source_offset,
                       .source_size = trace.source_size,
                       .source_words = trace.source_words,
                       .legalization = trace.legalization,
                       .copied_original = trace.copied_original,
                       .semantic_lowering = trace.semantic_lowering,
                       .changed = trace.changed,
                       .emitted_in_cave = false,
                       .target_offset = trace.target_offset + target_delta,
                       .target_words = trace.target_words});
    }
  };

  auto copy_original_instruction = [&](const Instruction &inst, uint64_t offset,
                                       std::vector<uint8_t> &kernel_text,
                                       std::vector<PendingTrace> &pending_traces) {
    const uint32_t inst_size = inst.size();
    const uint64_t target_offset = kernel_text.size();
    const auto *words = reinterpret_cast<const uint32_t *>(text.data() + offset);
    std::vector<uint32_t> copied_words(words, words + inst_size / sizeof(uint32_t));
    append_words(kernel_text, copied_words);
    // Continued-failure mode is diagnostic-only. Emit an explicit copy event so
    // diff reports make it clear which failed source instruction was preserved.
    queue_trace(pending_traces, inst, offset, nullptr, true, false, false, target_offset,
                std::move(copied_words));
  };

  auto continue_after_instruction_error = [&](const Instruction &inst, uint64_t offset,
                                              std::vector<uint8_t> &kernel_text,
                                              std::vector<PendingTrace> &pending_traces) {
    if (!continue_after_failure)
      return false;
    copy_original_instruction(inst, offset, kernel_text, pending_traces);
    return true;
  };

  auto write_words_at = [](std::vector<uint8_t> &dst, uint64_t offset,
                           std::span<const uint32_t> words) {
    if (words.empty())
      return;
    std::memcpy(dst.data() + offset, words.data(), words.size() * sizeof(uint32_t));
  };

  auto write_launch_stub = [&](KernelTextLayout &layout, uint64_t stub_offset,
                               uint64_t target_offset) {
    uint64_t cursor = stub_offset;
    write_words_at(translated_text, cursor, layout.translation->prologue_words);
    cursor += layout.translation->prologue_words.size() * sizeof(uint32_t);

    const auto branch_dwords = compute_sopp_branch_simm16(cursor, target_offset);
    assert(branch_dwords &&
           "kernarg preload launch stubs are synthesized adjacent to their relocated bodies");
    const uint32_t branch = build_s_branch(*branch_dwords, host_arch_);
    write_words_at(translated_text, cursor, std::span<const uint32_t>(&branch, 1));
  };

  auto relocation_diagnostic_kind = [](const TextRelocationResult &relocation) {
    if (relocation.message.find("range") != std::string::npos ||
        relocation.message.find("exceeds") != std::string::npos ||
        relocation.message.find("cannot encode") != std::string::npos ||
        relocation.message.find("outside translated") != std::string::npos) {
      return DiagnosticKind::ResourceLimit;
    }
    return DiagnosticKind::Legalization;
  };

  struct KernelFailure {
    DiagnosticKind kind = DiagnosticKind::Legalization;
    std::string message;
    std::optional<uint64_t> guest_offset;
    std::string mnemonic;
    std::vector<std::string> required_work;
  };

  auto make_kernel_failure = [](DiagnosticKind kind, std::string message,
                                std::optional<uint64_t> guest_offset = std::nullopt,
                                std::string mnemonic = {},
                                std::vector<std::string> required_work = {}) {
    return KernelFailure{kind, std::move(message), guest_offset, std::move(mnemonic),
                         std::move(required_work)};
  };

  auto emit_skipped_kernel = [&](const KernelTranslationScope &scope,
                                 KernelFailure failure) -> bool {
    assert(scope.translation != nullptr && "kernel scope should have descriptor translation");
    if (scope.blocks.empty()) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "cannot skip failed kernel without a decoded source block",
                   scope.translation->entry_text_offset);
      return false;
    }

    const uint64_t source_entry = scope.translation->entry_text_offset;
    // Keep the skipped symbol loadable without ever placing guest ISA bytes in
    // a target ELF. Some HIP module paths cache kernel handles without going
    // through the HSA symbol APIs that rocjitsu can reject, so the descriptor is
    // redirected to a target-ISA endpgm stub. If the skipped kernel is actually
    // dispatched, it becomes a no-op rather than executing invalid source
    // instructions on the host GPU; the skipped-kernel diagnostic remains the
    // authoritative error for that symbol.
    const uint64_t padding = padding_for_residue(translated_text.size(), source_entry % 256, 256);
    append_nop_padding(translated_text, padding, host_arch_);
    const uint64_t target_entry = translated_text.size();
    const uint32_t endpgm = build_s_endpgm(host_arch_);
    append_words(translated_text, std::span<const uint32_t>(&endpgm, 1));
    if (scope.translation->has_kernarg_preload) {
      append_nop_padding(translated_text, kKernargPreloadSkipBytes - sizeof(uint32_t), host_arch_);
      append_words(translated_text, std::span<const uint32_t>(&endpgm, 1));
    }

    for (KdTranslation &translation : descriptor_translations) {
      if (translation.entry_text_offset != source_entry)
        continue;
      translation.target_entry_text_offset = target_entry;
      translation.target_body_entry_text_offset = target_entry;
      translation.skipped = true;
      // A skipped descriptor must describe the target stub, not the failed
      // guest kernel. Leaving oversized SGPR/LDS/private requirements in place
      // can make HIP fail during launch even though the entry points at safe
      // target code. Granulated zero encodes the minimum allocation bucket.
      translation.target_vgpr_count = 0;
      translation.target_vgpr_allocation_count = 0;
      translation.target_vgpr_granulated = 0;
      translation.target_agpr_count = 0;
      translation.target_accvgpr_base = 4;
      translation.target_sgpr_count = 0;
      translation.target_sgpr_granulated = 0;
      translation.target_lds_size = 0;
      translation.lds_overflow_size = 0;
      translation.needs_lds_overflow_buf = false;
      translation.lds_overflow_base_sgpr = 0;
      translation.lds_overflow_prologue_temp_sgpr = 0;
      translation.lds_overflow_base_sgpr_spill_per_use = false;
      translation.lds_overflow_base_pointer_spilled = false;
      translation.lds_overflow_base_pointer_spill_offset = 0;
      translation.lds_overflow_entry_temp_vgprs_valid = false;
      translation.lds_overflow_entry_temp_vgpr_lo = 0;
      translation.lds_overflow_entry_temp_vgpr_hi = 0;
      translation.kernarg_size = 0;
      translation.target_kernarg_size = 0;
      translation.lds_overflow_kernarg_pointer_offset = 0;
      translation.lds_overflow_pointer_in_dispatch_packet = false;
      translation.target_private_size = 0;
      translation.target_user_sgpr_count = 0;
      translation.prologue_words.clear();
      if (!translation.kernel_name.empty() &&
          std::ranges::find(result.skipped_kernel_symbols, translation.kernel_name) ==
              result.skipped_kernel_symbols.end()) {
        result.skipped_kernel_symbols.push_back(translation.kernel_name);
      }
    }

    std::string message = "skipped kernel " + kernel_label(*scope.translation) +
                          " after translation error: " + std::move(failure.message);
    append_warning(result.diagnostics, DiagnosticKind::KernelSkipped, std::move(message),
                   failure.guest_offset ? failure.guest_offset
                                        : std::optional<uint64_t>(source_entry),
                   std::move(failure.mnemonic), std::move(failure.required_work));
    return true;
  };

  auto fail_or_skip_kernel = [&](const KernelTranslationScope &scope, KernelFailure failure,
                                 size_t output_begin,
                                 const std::vector<KdTranslation> &descriptor_snapshot) -> bool {
    if (!skip_failed_kernels) {
      append_error(result.diagnostics, failure.kind, std::move(failure.message),
                   failure.guest_offset, std::move(failure.mnemonic),
                   std::move(failure.required_work));
      return false;
    }

    const uint64_t source_entry = scope.translation->entry_text_offset;
    translated_text.resize(output_begin);
    if (descriptor_translations.size() != descriptor_snapshot.size()) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "descriptor snapshot size changed during skip rollback", source_entry);
      return false;
    }
    for (size_t i = 0; i < descriptor_translations.size(); ++i)
      descriptor_translations[i] = descriptor_snapshot[i];

    KernelTranslationScope restored_scope = scope;
    restored_scope.translation = nullptr;
    for (KdTranslation &translation : descriptor_translations) {
      if (!same_kernel_scope_variant(translation, *scope.translation))
        continue;
      restored_scope.translation = &translation;
      break;
    }
    if (restored_scope.translation == nullptr) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "failed kernel descriptor was lost during skip rollback", source_entry);
      return false;
    }

    return emit_skipped_kernel(restored_scope, std::move(failure));
  };

  auto reserve_long_branch_sgpr_pair = [&](TranslationContext &context) -> std::optional<uint16_t> {
    auto base = next_long_branch_sgpr_pair(context, host_arch_);
    if (!base)
      return std::nullopt;
    context.require_sgprs(static_cast<uint32_t>(*base) + 2);
    return base;
  };

  for (const KernelTranslationScope &scope : scopes) {
    if (scope.blocks.empty())
      continue;
    assert(scope.translation != nullptr && "kernel scope should have descriptor translation");
    if (scope.translation->skipped)
      continue;

    const size_t output_begin = translated_text.size();
    const std::vector<KdTranslation> descriptor_snapshot = descriptor_translations;
    bool skip_scope = false;

    if (!scope.translation->supported) {
      auto failure = make_kernel_failure(
          DiagnosticKind::KernelDescriptor,
          "kernel descriptor translation requires unsupported resource or ABI virtualization");
      for (const TranslationDiagnostic &diagnostic : scope.translation->diagnostics) {
        if (diagnostic.severity != DiagnosticSeverity::Error)
          continue;
        failure.kind = diagnostic.kind;
        failure.message = diagnostic.message;
        failure.guest_offset = diagnostic.guest_offset;
        failure.mnemonic = diagnostic.mnemonic;
        failure.required_work = diagnostic.required_work;
        break;
      }
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot))
        continue;
      return leave_unchanged();
    }

    // Phase 3: translate this kernel into a temporary, source-ordered body. The
    // body starts at offset zero while it is being built; after final padding and
    // any launch window are chosen, every recorded target offset is rebased into
    // the output .text. This lets instruction expansions change block sizes
    // without precomputing speculative side-region offsets.
    KernelTextLayout layout;
    layout.translation = scope.translation;
    layout.source_entry = scope.translation->entry_text_offset;
    const bool has_kernarg_preload = scope.translation->has_kernarg_preload;
    const uint64_t source_preload_entry = scope.translation->kernarg_preload_entry_text_offset;

    TranslationContext kernel_context(
        scope.translation->target_vgpr_count, scope.translation->target_agpr_count,
        scope.translation->target_accvgpr_base, scope.translation->target_sgpr_count,
        scope.translation->target_private_size);
    if (scope.translation->needs_lds_overflow_buf) {
      auto virtual_lds_base = reserve_virtual_lds_base_sgpr_pair(
          kernel_context, KernelBlockScope(scope.blocks), host_arch_);
      if (!virtual_lds_base) {
        auto failure = make_kernel_failure(
            DiagnosticKind::ResourceLimit,
            "virtual LDS lowering cannot reserve a backing-buffer SGPR pair", layout.source_entry);
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot))
          continue;
        return leave_unchanged();
      }
      kernel_context.virtualize_lds = true;
      kernel_context.virtual_lds_base_sgpr = virtual_lds_base->base;
      kernel_context.virtual_lds_base_sgpr_spill_per_use = virtual_lds_base->spill_per_use;
      kernel_context.virtual_lds_kernarg_segment_ptr_sgpr =
          scope.translation->lds_overflow_pointer_in_dispatch_packet
              ? scope.translation->dispatch_ptr_sgpr
              : scope.translation->kernarg_segment_ptr_sgpr;
      kernel_context.virtual_lds_kernarg_pointer_offset =
          scope.translation->lds_overflow_kernarg_pointer_offset;
      scope.translation->lds_overflow_base_sgpr = virtual_lds_base->base;
      scope.translation->lds_overflow_prologue_temp_sgpr = virtual_lds_base->prologue_temp;
      scope.translation->lds_overflow_base_sgpr_spill_per_use = virtual_lds_base->spill_per_use;
      if (virtual_lds_base->spill_per_use) {
        const uint32_t pointer_spill = kernel_context.reserve_persistent_semantic_spill_dwords(2);
        kernel_context.virtual_lds_base_pointer_spilled = true;
        kernel_context.virtual_lds_base_pointer_spill_offset = pointer_spill;
        scope.translation->lds_overflow_base_pointer_spilled = true;
        scope.translation->lds_overflow_base_pointer_spill_offset = pointer_spill;
      }
      if (!append_virtual_lds_entry_prologue(*scope.translation)) {
        auto failure = make_kernel_failure(
            DiagnosticKind::KernelDescriptor,
            "virtual LDS lowering cannot materialize backing-buffer pointer entry prologue",
            layout.source_entry);
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot))
          continue;
        return leave_unchanged();
      }
    }

    const uint64_t prologue_bytes = scope.translation->prologue_words.size() * sizeof(uint32_t);
    const uint64_t launch_stub_bytes = prologue_bytes + sizeof(uint32_t);
    if (has_kernarg_preload && launch_stub_bytes > kKernargPreloadSkipBytes) {
      auto failure = make_kernel_failure(
          DiagnosticKind::KernelDescriptor,
          "kernel descriptor prologue does not fit in the 256-byte kernarg preload compatibility "
          "window",
          layout.source_entry);
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot))
        continue;
      return leave_unchanged();
    }
    const bool can_use_long_direct_branches =
        next_long_branch_sgpr_pair(kernel_context, host_arch_).has_value();
    LivenessAnalysisOptions liveness_options;
    if (options_.debug_min_free_vgpr)
      liveness_options.min_free_vgpr = *options_.debug_min_free_vgpr;
    const auto liveness_edges = scoped_call_liveness_edges(KernelBlockScope(scope.blocks), text);
    LivenessAnalysis liveness(KernelBlockScope(scope.blocks), liveness_options, liveness_edges);

    // Phase 4: translate each relocated body instruction at the current cursor.
    // Return-like s_setpc_b64 instructions are accepted only when they are the
    // terminator of a block reached from a validated call edge in this
    // kernel-local scope. Recovered indirect setpc/swappc consumers reserve a
    // fixed maximum-size window when recovery proves one effective target. When
    // one dynamic consumer has multiple recovered targets, no single direct
    // window can preserve semantics; DBT keeps the original indirect consumer
    // and asks the patch layer to rewrite each source-side PC builder once.
    const std::unordered_set<uint64_t> valid_call_return_offsets =
        scoped_call_return_offsets(KernelBlockScope(scope.blocks), text);
    struct RecoveredConsumer {
      std::vector<IndirectCallFixup> fixups;
      bool use_transfer_window = false;
      IndirectCallFixup window_fixup;
    };
    std::unordered_map<uint64_t, RecoveredConsumer> recovered_indirect_by_call;
    for (BasicBlock *block : scope.blocks) {
      for (const IndirectCallFixup &source_fixup : block->static_indirect_call_fixups()) {
        recovered_indirect_by_call[source_fixup.source_call_offset].fixups.push_back(source_fixup);
      }
    }

    std::vector<IndirectCallFixup> pending_builder_fixups;
    for (auto &[source_call_offset, consumer] : recovered_indirect_by_call) {
      if (consumer.fixups.empty())
        continue;

      const IndirectCallFixup &first = consumer.fixups.front();
      bool single_effective_target = true;
      for (const IndirectCallFixup &fixup : consumer.fixups) {
        if (fixup.source_call_sreg != first.source_call_sreg ||
            fixup.source_is_call != first.source_is_call ||
            fixup.source_return_sreg != first.source_return_sreg) {
          auto failure =
              make_kernel_failure(DiagnosticKind::Legalization,
                                  "recovered indirect branch has inconsistent consumer metadata",
                                  source_call_offset, "indirect branch");
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }
        if (fixup.source_target_offset != first.source_target_offset)
          single_effective_target = false;
      }
      if (skip_scope)
        break;

      if (single_effective_target) {
        consumer.use_transfer_window = true;
        consumer.window_fixup = first;
      } else {
        pending_builder_fixups.insert(pending_builder_fixups.end(), consumer.fixups.begin(),
                                      consumer.fixups.end());
      }
    }
    if (skip_scope)
      continue;

    std::vector<uint8_t> kernel_text;
    std::vector<PendingTrace> pending_traces;
    uint64_t source_body_size = 0;
    for (BasicBlock *block : scope.blocks)
      source_body_size += block->size();
    const uint64_t recovered_window_growth =
        recovered_indirect_by_call.size() * kMaxRecoveredIndirectTransferWords * sizeof(uint32_t);
    kernel_text.reserve(static_cast<size_t>(std::min<uint64_t>(
        source_body_size + recovered_window_growth, std::numeric_limits<size_t>::max())));

    std::unordered_set<uint64_t> needed_builder_source_offsets;
    needed_builder_source_offsets.reserve(pending_builder_fixups.size() * 3);
    for (const IndirectCallFixup &fixup : pending_builder_fixups) {
      needed_builder_source_offsets.insert(fixup.source_getpc_offset);
      needed_builder_source_offsets.insert(fixup.source_recovery_begin_offset);
      needed_builder_source_offsets.insert(fixup.source_recovery_end_offset);
    }
    std::unordered_map<uint64_t, uint64_t> target_offset_by_source_offset;
    target_offset_by_source_offset.reserve(needed_builder_source_offsets.size());
    layout.body_begin = 0;
    layout.blocks.reserve(scope.blocks.size());
    uint64_t next_branch_island_pool_offset = kDirectBranchIslandSpacingBytes;
    for (BasicBlock *block : scope.blocks) {
      BlockPlacement placement{.block = block,
                               .source_start = block->start_offset(),
                               .source_end = block->end_offset(),
                               .target_start = kernel_text.size(),
                               .target_end = kernel_text.size()};

      for (auto it = block->instructions().begin(); it != block->instructions().end(); ++it) {
        const auto &inst = *it;
        const uint64_t offset = inst.src_loc();
        const uint64_t target_offset = kernel_text.size();
        const uint32_t inst_size = inst.size();
        if (needed_builder_source_offsets.contains(offset))
          target_offset_by_source_offset.emplace(offset, target_offset);

        const auto recovered_it = recovered_indirect_by_call.find(offset);
        const bool has_recovered_indirect_call = recovered_it != recovered_indirect_by_call.end();
        const bool recovered_indirect_return = valid_call_return_offsets.contains(offset);
        const auto direct_branch_delta = inst.branch_offset_bytes();
        if ((inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0 &&
            !has_recovered_indirect_call && !recovered_indirect_return && !direct_branch_delta) {
          auto failure = make_kernel_failure(
              DiagnosticKind::Legalization,
              "indirect branch or call target recovery is not implemented for relocated kernel "
              "text",
              offset, std::string(inst.mnemonic()));
          if (continue_after_failure && !skip_failed_kernels) {
            append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                         failure.mnemonic, failure.required_work);
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
              continue;
            }
          }
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        if (direct_branch_delta) {
          // Record direct branches while emitting the body, but patch only after
          // every block has a final target placement. This keeps fallthrough
          // implicit and limits fixups to explicit PC-relative edges. Emit the
          // branch into a fixed-size patch window. Kernels with a legal
          // descriptor-backed SGPR pair reserve the long form up front; kernels
          // already at the SGPR allocation limit keep compact branch slots so
          // DBT does not create artificial range pressure it cannot repair.
          const int64_t source_target =
              static_cast<int64_t>(offset + inst_size) + static_cast<int64_t>(*direct_branch_delta);
          if (source_target < 0) {
            auto failure =
                make_kernel_failure(DiagnosticKind::Legalization,
                                    "direct branch target is outside the source .text range",
                                    offset, std::string(inst.mnemonic()));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }
          uint64_t branch_window_bytes = inst.size();
          if (can_use_long_direct_branches) {
            branch_window_bytes = kMaxDirectBranchTransferWords * sizeof(uint32_t);
          } else if ((inst.flags() & COND_BRANCH) != 0 &&
                     supports_direct_branch_island_window(inst)) {
            branch_window_bytes = 2 * sizeof(uint32_t);
          }
          layout.branch_fixups.push_back(
              {.inst = &inst,
               .source_inst_offset = offset,
               .source_target_offset = static_cast<uint64_t>(source_target),
               .target_inst_offset = target_offset,
               .target_window_bytes = branch_window_bytes});

          if (!inst.raw_encoding()) {
            auto failure = make_kernel_failure(DiagnosticKind::Legalization,
                                               "direct branch is missing raw encoding", offset,
                                               std::string(inst.mnemonic()));
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }

          const InstructionLegalization *branch_leg = nullptr;
          if (legalization_lookup_)
            branch_leg = legalization_lookup_(inst.encoding_id(), inst.opcode());
          const uint16_t branch_dst_opcode = branch_leg ? branch_leg->target_opcode : inst.opcode();

          bool copied_original = false;
          bool changed = false;
          std::vector<uint32_t> target_words;
          if (!handle_encoding(inst, offset, kernel_text, branch_dst_opcode, text,
                               trace_callback_ != nullptr, copied_original, changed,
                               target_words)) {
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces))
              continue;
            return leave_unchanged();
          }
          append_nop_padding(kernel_text, branch_window_bytes - inst.size(), host_arch_);
          queue_trace(pending_traces, inst, offset, branch_leg, copied_original, false, changed,
                      target_offset, std::move(target_words));
          continue;
        }

        if (has_recovered_indirect_call && recovered_it->second.use_transfer_window) {
          const IndirectCallFixup &source_fixup = recovered_it->second.window_fixup;
          layout.recovered_indirect_fixups.push_back(
              {.source_call_offset = source_fixup.source_call_offset,
               .source_target_offset = source_fixup.source_target_offset,
               .target_window_offset = target_offset,
               .target_sreg = source_fixup.source_call_sreg,
               .return_sreg = source_fixup.source_return_sreg,
               .is_call = source_fixup.source_is_call});
          append_nop_padding(kernel_text, kMaxRecoveredIndirectTransferWords * sizeof(uint32_t),
                             host_arch_);
          continue;
        }

        const uint32_t *raw = inst.raw_encoding();
        if (!raw) {
          copy_original_instruction(inst, offset, kernel_text, pending_traces);
          continue;
        }

        const InstructionLegalization *leg = nullptr;
        if (legalization_lookup_)
          leg = legalization_lookup_(inst.encoding_id(), inst.opcode());

        const uint16_t dst_opcode = leg ? leg->target_opcode : inst.opcode();

        // Try semantic lowering before raw encoding translation. A matched
        // semantic rule that cannot safely emit code is a translation error:
        // falling through would silently preserve guest semantics on the wrong
        // host ISA.
        {
          auto expansion =
              semantic_translator_->try_lower_expand(inst, offset, liveness, kernel_context);
          if (expansion.status == ExpandStatus::Failed) {
            auto failure = make_kernel_failure(
                DiagnosticKind::ExpandFailed,
                expansion.message.empty()
                    ? "semantic EXPAND rule matched, but could not safely lower"
                    : expansion.message,
                offset, std::string(inst.mnemonic()), std::move(expansion.required_work));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }

          if (expansion.status == ExpandStatus::Success) {
            std::vector<uint32_t> target_words = std::move(expansion.words);
            append_words(kernel_text, target_words);
            queue_trace(pending_traces, inst, offset, leg, false, true, true, target_offset,
                        std::move(target_words));
            continue;
          }
        }

        {
          auto virtual_lds_expansion = lower_virtual_lds_ds_instruction(inst, kernel_context);
          if (virtual_lds_expansion.status == ExpandStatus::Failed) {
            auto failure = make_kernel_failure(DiagnosticKind::ExpandFailed,
                                               virtual_lds_expansion.message.empty()
                                                   ? "virtual LDS lowering failed"
                                                   : virtual_lds_expansion.message,
                                               offset, std::string(inst.mnemonic()),
                                               std::move(virtual_lds_expansion.required_work));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }

          if (virtual_lds_expansion.status == ExpandStatus::Success) {
            std::vector<uint32_t> target_words = std::move(virtual_lds_expansion.words);
            append_words(kernel_text, target_words);
            queue_trace(pending_traces, inst, offset, leg, false, true, true, target_offset,
                        std::move(target_words));
            continue;
          }
        }

        if (leg && leg->action == Action::Expand) {
          auto failure = make_kernel_failure(
              DiagnosticKind::ExpandMissing,
              "legalization requires EXPAND, but no expansion rule is implemented", offset,
              std::string(inst.mnemonic()), {"Add a semantic expansion rule for this mnemonic."});
          if (continue_after_failure && !skip_failed_kernels) {
            append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                         failure.mnemonic, failure.required_work);
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
              continue;
            }
          }
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        bool copied_original = false;
        bool changed = false;
        std::vector<uint32_t> target_words;
        if (!handle_encoding(inst, offset, kernel_text, dst_opcode, text,
                             trace_callback_ != nullptr, copied_original, changed, target_words)) {
          if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
            continue;
          }
          return leave_unchanged();
        }
        queue_trace(pending_traces, inst, offset, leg, copied_original, false, changed,
                    target_offset, std::move(target_words));
      }
      if (skip_scope)
        break;
      placement.target_end = kernel_text.size();
      layout.blocks.push_back(placement);
      if (needed_builder_source_offsets.contains(block->end_offset()))
        target_offset_by_source_offset.emplace(block->end_offset(), kernel_text.size());
      if (!can_use_long_direct_branches && block != scope.blocks.back() &&
          kernel_text.size() >= next_branch_island_pool_offset) {
        append_direct_branch_island_pool(kernel_text, layout, host_arch_);
        next_branch_island_pool_offset = kernel_text.size() + kDirectBranchIslandSpacingBytes;
      }
    }
    if (skip_scope)
      continue;
    layout.body_end = kernel_text.size();

    if (continue_after_failure && has_error_diagnostic(result.diagnostics))
      continue;

    for (IndirectCallFixup fixup : pending_builder_fixups) {
      const auto getpc_it = target_offset_by_source_offset.find(fixup.source_getpc_offset);
      const auto begin_it = target_offset_by_source_offset.find(fixup.source_recovery_begin_offset);
      const auto end_it = target_offset_by_source_offset.find(fixup.source_recovery_end_offset);
      if (getpc_it == target_offset_by_source_offset.end() ||
          begin_it == target_offset_by_source_offset.end() ||
          end_it == target_offset_by_source_offset.end()) {
        auto failure = make_kernel_failure(
            DiagnosticKind::Legalization,
            "recovered indirect branch builder is not fully present in the relocated body",
            fixup.source_call_offset, "indirect branch");
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot)) {
          skip_scope = true;
          break;
        }
        return leave_unchanged();
      }

      fixup.target_getpc_offset = getpc_it->second;
      fixup.target_recovery_begin_offset = begin_it->second;
      fixup.target_recovery_end_offset = end_it->second;
      layout.recovered_builder_fixups.push_back(fixup);
    }
    if (skip_scope)
      continue;

    auto body_entry = target_for_source_offset(layout, layout.source_entry);
    if (!body_entry) {
      auto failure =
          make_kernel_failure(DiagnosticKind::KernelDescriptor,
                              "kernel descriptor entry offset is not present in the relocated body",
                              layout.source_entry);
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot))
        continue;
      return leave_unchanged();
    }
    layout.target_body_entry = *body_entry;

    std::optional<uint64_t> preload_body_entry;
    if (has_kernarg_preload) {
      preload_body_entry = target_for_source_offset(layout, source_preload_entry);
      if (!preload_body_entry) {
        auto failure = make_kernel_failure(
            DiagnosticKind::KernelDescriptor,
            "kernarg preload firmware entry offset is not present in the relocated body",
            source_preload_entry);
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot))
          continue;
        return leave_unchanged();
      }
    }

    uint64_t target_delta = 0;
    if (has_kernarg_preload) {
      // Kernarg-preload kernels have two hardware-visible entries separated by
      // exactly 256 bytes. Reserve that launch window before appending the body;
      // the stubs are written after the body offsets have been rebased.
      const uint64_t launch_padding =
          padding_for_residue(translated_text.size(), layout.source_entry % 256, 256);
      append_nop_padding(translated_text, launch_padding, host_arch_);
      layout.target_entry = translated_text.size();
      const uint64_t launch_end =
          layout.target_entry + kKernargPreloadSkipBytes + launch_stub_bytes;
      append_nop_padding(translated_text, launch_end - translated_text.size(), host_arch_);
      target_delta = translated_text.size();
    } else {
      const uint64_t body_padding = padding_for_residue(
          translated_text.size() + layout.target_body_entry, layout.source_entry % 256, 256);
      append_nop_padding(translated_text, body_padding, host_arch_);
      target_delta = translated_text.size();
    }

    rebase_kernel_text_layout(layout, target_delta);
    translated_text.insert(translated_text.end(), kernel_text.begin(), kernel_text.end());

    if (has_kernarg_preload) {
      assert(preload_body_entry && "preload body entry was checked before rebase");
      write_launch_stub(layout, layout.target_entry, layout.target_body_entry);
      write_launch_stub(layout, layout.target_entry + kKernargPreloadSkipBytes,
                        *preload_body_entry + target_delta);
    } else if (!scope.translation->prologue_words.empty()) {
      // Descriptor prologues are hardware entry points. Align the prologue to the
      // original entry residue, then branch into the relocated body. The prologue
      // lives in .text but is not part of the source block placement map.
      const uint64_t prologue_padding =
          padding_for_residue(translated_text.size(), layout.source_entry % 256, 256);
      append_nop_padding(translated_text, prologue_padding, host_arch_);
      layout.target_entry = translated_text.size();
      append_words(translated_text, scope.translation->prologue_words);

      const uint64_t branch_pc = translated_text.size();
      const auto branch_dwords = compute_sopp_branch_simm16(branch_pc, layout.target_body_entry);
      if (!branch_dwords) {
        auto failure = make_kernel_failure(
            DiagnosticKind::ResourceLimit,
            "kernel descriptor prologue branch range exceeds s_branch simm16", layout.source_entry);
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot))
          continue;
        return leave_unchanged();
      }
      const uint32_t branch = build_s_branch(*branch_dwords, host_arch_);
      append_words(translated_text, std::span<const uint32_t>(&branch, 1));
    } else {
      layout.target_entry = layout.target_body_entry;
    }

    // Phase 5: now that every emitted source block has a final target offset,
    // patch explicit direct branches, recovered source-side builders, and
    // recovered indirect transfer windows.
    auto patched_direct_branches = patch_direct_branch_fixups(translated_text, layout, host_arch_);
    if (!patched_direct_branches.ok &&
        patched_direct_branches.message.find("exceeds encoded branch range") != std::string::npos) {
      if (auto sgpr = reserve_long_branch_sgpr_pair(kernel_context)) {
        layout.long_branch_sgpr = *sgpr;
        patched_direct_branches = patch_direct_branch_fixups(translated_text, layout, host_arch_);
      }
    }
    if (!patched_direct_branches.ok) {
      auto failure = make_kernel_failure(relocation_diagnostic_kind(patched_direct_branches),
                                         patched_direct_branches.message,
                                         patched_direct_branches.source_offset);
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot))
        continue;
      return leave_unchanged();
    }

    if (auto patched = patch_recovered_builder_fixups(translated_text, layout, host_arch_);
        !patched.ok) {
      auto failure = make_kernel_failure(relocation_diagnostic_kind(patched), patched.message,
                                         patched.source_offset, "indirect branch");
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot))
        continue;
      return leave_unchanged();
    }

    if (auto patched = patch_recovered_indirect_fixups(translated_text, layout, host_arch_);
        !patched.ok) {
      auto failure = make_kernel_failure(relocation_diagnostic_kind(patched), patched.message,
                                         patched.source_offset, "indirect branch");
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot))
        continue;
      return leave_unchanged();
    }

    if (kernel_context.required_vgpr_count > kernel_context.num_vgprs)
      scope.translation->target_vgpr_count = kernel_context.required_vgpr_count;
    if (kernel_context.required_sgpr_count > kernel_context.num_sgprs)
      scope.translation->target_sgpr_count = kernel_context.required_sgpr_count;
    if (kernel_context.required_private_segment_fixed_size >
        kernel_context.private_segment_fixed_size)
      scope.translation->target_private_size = kernel_context.required_private_segment_fixed_size;

    if (scope.translation->target_vgpr_count != kernel_context.num_vgprs ||
        scope.translation->target_sgpr_count != kernel_context.num_sgprs ||
        scope.translation->target_private_size != kernel_context.private_segment_fixed_size) {
      // Semantic rules may allocate descriptor-backed scratch registers or
      // per-lane private spill slots beyond the kernel's original resources.
      // Recompute the descriptor with those larger minimums before patching it
      // into the output image.
      KernelDescriptorTranslationOptions descriptor_options;
      descriptor_options.minimum_vgprs = scope.translation->target_vgpr_count;
      descriptor_options.minimum_sgprs = scope.translation->target_sgpr_count;
      descriptor_options.private_segment_fixed_size_addend =
          scope.translation->target_private_size - kernel_context.private_segment_fixed_size;
      descriptor_options.virtualize_lds = scope.translation->needs_lds_overflow_buf;
      descriptor_options.allow_oversized_lds =
          can_emit_virtual_lds_variants && !scope.translation->needs_lds_overflow_buf;

      // Descriptor growth is intentionally done after instruction lowering so
      // each kernel is translated once. Only descriptors that enter this code
      // scope need the larger register counts; rescanning the whole image would
      // also recompute unrelated kernels and risks mixing diagnostics across
      // scopes.
      bool recomputed_descriptor = false;
      for (KdTranslation &translation : descriptor_translations) {
        if (!same_kernel_scope_variant(translation, *scope.translation))
          continue;

        auto updated = descriptor_translator.translate_descriptor(
            patcher.image_bytes(), translation.descriptor_file_offset,
            translation.entry_text_offset, descriptor_options);
        if (!updated) {
          auto failure =
              make_kernel_failure(DiagnosticKind::KernelDescriptor,
                                  "kernel descriptor translation could not be recomputed");
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }
        updated->kernel_name = translation.kernel_name;
        updated->virtual_lds_variant = translation.virtual_lds_variant;
        updated->lds_overflow_base_sgpr = translation.lds_overflow_base_sgpr;
        updated->lds_overflow_prologue_temp_sgpr = translation.lds_overflow_prologue_temp_sgpr;
        updated->lds_overflow_base_sgpr_spill_per_use =
            translation.lds_overflow_base_sgpr_spill_per_use;
        updated->lds_overflow_base_pointer_spilled = translation.lds_overflow_base_pointer_spilled;
        updated->lds_overflow_base_pointer_spill_offset =
            translation.lds_overflow_base_pointer_spill_offset;
        updated->lds_overflow_entry_temp_vgprs_valid =
            translation.lds_overflow_entry_temp_vgprs_valid;
        updated->lds_overflow_entry_temp_vgpr_lo = translation.lds_overflow_entry_temp_vgpr_lo;
        updated->lds_overflow_entry_temp_vgpr_hi = translation.lds_overflow_entry_temp_vgpr_hi;
        if (!append_virtual_lds_entry_prologue(*updated)) {
          auto failure = make_kernel_failure(
              DiagnosticKind::KernelDescriptor,
              "virtual LDS lowering cannot materialize backing-buffer pointer entry prologue");
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        if (!updated->supported) {
          if (skip_failed_kernels) {
            auto failure = make_kernel_failure(DiagnosticKind::KernelDescriptor,
                                               "kernel descriptor translation requires unsupported "
                                               "resource or ABI virtualization");
            for (const TranslationDiagnostic &diagnostic : updated->diagnostics) {
              if (diagnostic.severity != DiagnosticSeverity::Error)
                continue;
              failure.kind = diagnostic.kind;
              failure.message = diagnostic.message;
              failure.guest_offset = diagnostic.guest_offset;
              failure.mnemonic = diagnostic.mnemonic;
              failure.required_work = diagnostic.required_work;
              break;
            }
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot)) {
              skip_scope = true;
              break;
            }
          }
          append_diagnostics(result.diagnostics, updated->diagnostics);
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernel descriptor translation requires unsupported resource or ABI "
                       "virtualization; leaving code object unchanged");
          return leave_unchanged();
        }
        append_diagnostics(result.diagnostics, updated->diagnostics);

        if (updated->prologue_words != translation.prologue_words) {
          auto failure = make_kernel_failure(
              DiagnosticKind::KernelDescriptor,
              "kernel descriptor prologue changed after relocated text was emitted");
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        updated->target_entry_text_offset = layout.target_entry;
        updated->target_body_entry_text_offset = layout.target_body_entry;
        translation = std::move(*updated);
        recomputed_descriptor = true;
      }
      if (skip_scope)
        continue;

      if (!recomputed_descriptor) {
        auto failure = make_kernel_failure(DiagnosticKind::KernelDescriptor,
                                           "kernel descriptor translation could not be recomputed");
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot))
          continue;
        return leave_unchanged();
      }
    }

    flush_traces(pending_traces, target_delta);

    for (KdTranslation &translation : descriptor_translations) {
      if (!same_kernel_scope_variant(translation, *scope.translation))
        continue;
      translation.target_entry_text_offset = layout.target_entry;
      translation.target_body_entry_text_offset = layout.target_body_entry;
    }
  }

  if (continue_after_failure && has_error_diagnostic(result.diagnostics))
    return leave_unchanged();

  // Phase 6: write the relocated .text and descriptor entry offsets into the ELF.
  // Reachability-driven emission intentionally drops source padding and other
  // unreachable bytes. Keep ELF mutation one-sided by padding the relocated
  // .text back to at least the original size.
  if (translated_text.size() < text.size())
    append_nop_padding(translated_text, text.size() - translated_text.size(), host_arch_);

  std::unordered_set<uint64_t> applied_descriptors;
  for (const KdTranslation &translation : descriptor_translations) {
    if (translation.virtual_lds_variant)
      continue;
    if (applied_descriptors.insert(translation.descriptor_file_offset).second) {
      if (translation.skipped) {
        if (!patcher.apply_kernel_descriptor_translation(translation, host_arch_)) {
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "skipped kernel descriptor could not be patched to a target stub safely; "
                       "leaving code object unchanged");
          return leave_unchanged();
        }
        continue;
      }

      if (!patcher.apply_kernel_descriptor_translation(translation, host_arch_)) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "kernel descriptor translation could not be applied safely; leaving code "
                     "object unchanged");
        return leave_unchanged();
      }
    }
  }

  if (!patcher.replace_text(translated_text)) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "relocated .text could not be materialized safely; leaving code object unchanged");
    return leave_unchanged();
  }

  std::vector<uint64_t> virtual_descriptor_vaddrs(descriptor_translations.size(), 0);
  std::vector<KdTranslation> virtual_descriptor_translations;
  std::vector<size_t> virtual_descriptor_indices;
  for (size_t i = 0; i < descriptor_translations.size(); ++i) {
    const KdTranslation &translation = descriptor_translations[i];
    if (!translation.virtual_lds_variant || !translation.needs_lds_overflow_buf ||
        translation.skipped)
      continue;
    virtual_descriptor_translations.push_back(translation);
    virtual_descriptor_indices.push_back(i);
  }
  if (!virtual_descriptor_translations.empty()) {
    auto appended = patcher.append_kernel_descriptor_translations(virtual_descriptor_translations,
                                                                  host_arch_, 64);
    if (!appended || appended->size() != virtual_descriptor_translations.size()) {
      append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                   "virtual LDS sidecar descriptors could not be materialized safely; leaving code "
                   "object unchanged");
      return leave_unchanged();
    }
    for (size_t i = 0; i < appended->size(); ++i)
      virtual_descriptor_vaddrs[virtual_descriptor_indices[i]] = (*appended)[i].vaddr;
  }

  const auto patched_image = patcher.image_bytes();
  AmdGpuCodeObject patched_layout(patched_image.data(), patched_image.size());
  if (!patched_layout.is_valid()) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "relocated ELF could not be reparsed for runtime metadata; leaving code object "
                 "unchanged");
    return leave_unchanged();
  }

  std::vector<VirtualLdsKernelMetadata> virtual_lds_metadata;
  for (const KdTranslation &translation : descriptor_translations) {
    if (!translation.needs_lds_overflow_buf || translation.skipped)
      continue;
    const uint64_t descriptor_vaddr =
        patched_layout.kernel_descriptor_offset(translation.kernel_name);
    if (descriptor_vaddr == 0) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS metadata could not find the translated kernel descriptor symbol; "
                   "leaving code object unchanged");
      return leave_unchanged();
    }
    VirtualLdsKernelMetadata record{};
    record.kernel_name = translation.kernel_name;
    record.normal_descriptor_vaddr = descriptor_vaddr;
    if (translation.virtual_lds_variant) {
      const size_t index = static_cast<size_t>(&translation - descriptor_translations.data());
      record.virtual_descriptor_vaddr = virtual_descriptor_vaddrs[index];
      if (record.virtual_descriptor_vaddr == 0) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "virtual LDS metadata could not find the appended sidecar descriptor; leaving "
                     "code object unchanged");
        return leave_unchanged();
      }
    } else {
      record.virtual_descriptor_vaddr = descriptor_vaddr;
    }
    record.static_lds_bytes = translation.lds_overflow_size;
    record.kernarg_size = translation.kernarg_size;
    record.backing_pointer_kernarg_offset = translation.lds_overflow_kernarg_pointer_offset;
    record.virtual_lds_base_sgpr = translation.lds_overflow_base_sgpr;
    record.flags |= kVirtualLdsFlagRuntimeStateBlock;
    if (translation.lds_overflow_pointer_in_dispatch_packet)
      record.flags |= kVirtualLdsFlagBackingPointerInDispatchPacket;
    if (translation.workgroup_id_sgpr_x >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdX;
    if (translation.workgroup_id_sgpr_y >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdY;
    if (translation.workgroup_id_sgpr_z >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdZ;
    if (translation.virtual_lds_variant) {
      if (record.normal_descriptor_vaddr >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          record.virtual_descriptor_vaddr >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "virtual LDS descriptor addresses cannot be encoded in dispatch metadata; "
                     "leaving code object unchanged");
        return leave_unchanged();
      }
      const int64_t normal_vaddr = static_cast<int64_t>(record.normal_descriptor_vaddr);
      const int64_t virtual_vaddr = static_cast<int64_t>(record.virtual_descriptor_vaddr);
      VirtualLdsDescriptorDispatchMetadata dispatch_metadata{
          .virtual_descriptor_delta = virtual_vaddr - normal_vaddr,
          .kernarg_size = translation.kernarg_size,
          .backing_pointer_kernarg_offset = translation.lds_overflow_kernarg_pointer_offset,
          .flags = record.flags,
      };
      if (!patcher.annotate_virtual_lds_descriptor_by_vaddr(record.normal_descriptor_vaddr,
                                                            dispatch_metadata)) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "virtual LDS descriptor dispatch metadata could not be written; leaving code "
                     "object unchanged");
        return leave_unchanged();
      }
    }
    virtual_lds_metadata.push_back(std::move(record));
  }
  if (!virtual_lds_metadata.empty()) {
    const auto metadata_bytes = serialize_virtual_lds_metadata(virtual_lds_metadata);
    if (metadata_bytes.empty() ||
        !patcher.append_nonalloc_section(kVirtualLdsMetadataSectionName, metadata_bytes, 8)) {
      append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                   "virtual LDS metadata could not be materialized safely; leaving code object "
                   "unchanged");
      return leave_unchanged();
    }
  }

  if (target_mach_)
    patcher.update_elf_flags(target_mach_);

  diagnostics_ = nullptr;
  result.elf_bytes = std::move(patcher).emit();
  return result;
}

bool BinaryTranslator::handle_encoding(const Instruction &inst, uint64_t offset,
                                       std::vector<uint8_t> &text, uint16_t dst_opcode,
                                       std::span<const uint8_t> orig_text, bool collect_trace_words,
                                       bool &copied_original, bool &changed,
                                       std::vector<uint32_t> &target_words) {
  const uint32_t *raw = inst.raw_encoding();
  assert(raw && "handle_encoding called without raw encoding");
  copied_original = false;
  changed = false;
  if (collect_trace_words)
    target_words.clear();

  if (!encoding_translate_) {
    copied_original = true;
    const size_t word_count = inst.size() / sizeof(uint32_t);
    if (collect_trace_words)
      target_words.assign(raw, raw + word_count);
    append_words(text, std::span<const uint32_t>(raw, word_count));
    return true;
  }

  const uint32_t w0 = raw[0];
  const uint32_t w1 = inst.size() > 4 ? raw[1] : 0;
  const uint32_t w2 = inst.size() > 8 ? raw[2] : 0;

  auto tr = encoding_translate_(inst.encoding_id(), w0, w1, w2, dst_opcode);

  if (tr.word_count == 0) {
    copied_original = true;
    const size_t word_count = inst.size() / sizeof(uint32_t);
    if (collect_trace_words)
      target_words.assign(raw, raw + word_count);
    append_words(text, std::span<const uint32_t>(raw, word_count));
    return true;
  }

  // Append trailing literal constant when the source instruction is larger
  // than the translated encoding. This handles single-word formats (SOP1,
  // SOP2, VOP1, VOP2, etc.) with a 32-bit literal appended when a source
  // operand is 0xFF. The encoding translator returns the format's native
  // word count; the literal is always one extra word beyond that.
  // Guard: only append if the gap is exactly one word (the literal). Larger
  // gaps would indicate a format mismatch, not a trailing literal.
  const uint32_t translated_bytes = tr.word_count * 4u;
  const uint32_t orig_bytes = inst.size();
  if (orig_bytes - translated_bytes == 4 && tr.word_count < 3) {
    uint32_t lit_word;
    std::memcpy(&lit_word, orig_text.data() + offset + translated_bytes, 4);
    tr.words[tr.word_count++] = lit_word;
  }

  append_words(text, std::span<const uint32_t>(tr.words, tr.word_count));
  if (collect_trace_words) {
    target_words.assign(tr.words, tr.words + tr.word_count);
    changed = words_changed(raw_words_for_inst(inst), target_words);
  }
  return true;
}

} // namespace rocjitsu
