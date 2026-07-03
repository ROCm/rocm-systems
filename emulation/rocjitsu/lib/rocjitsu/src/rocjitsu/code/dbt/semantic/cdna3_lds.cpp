// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna3_lds.cpp
/// @brief CDNA3 emission helpers for virtualizing LDS accesses through GLOBAL memory.

#include "rocjitsu/code/dbt/semantic/cdna3_lds.h"

#include "rocjitsu/code/dbt/semantic/cdna3_scratch.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"

#include <cstring>
#include <limits>
#include <utility>

namespace rocjitsu {
namespace {

inline constexpr uint8_t kCdna3ScalarNull = 0x7F;

[[nodiscard]] constexpr uint16_t vgpr_src(uint8_t reg) { return static_cast<uint16_t>(256u + reg); }

[[nodiscard]] uint32_t build_cdna3_vop1(uint16_t op, uint8_t vdst, uint16_t src0) {
  cdna3::Vop1MachineInst dst{};
  dst.encoding = 0x3F;
  dst.op = op & 0xFF;
  dst.vdst = vdst;
  dst.src0 = src0 & 0x1FF;

  uint32_t word = 0;
  std::memcpy(&word, &dst, sizeof(word));
  return word;
}

[[nodiscard]] uint32_t build_cdna3_v_mov_b32(uint8_t vdst, uint16_t src0) {
  return build_cdna3_vop1(cdna3::kVMovB32Vop1, vdst, src0);
}

[[nodiscard]] uint32_t build_cdna3_v_readfirstlane_b32(uint8_t sdst, uint8_t vsrc) {
  return build_cdna3_vop1(cdna3::kVReadfirstlaneB32Vop1, sdst, vgpr_src(vsrc));
}

[[nodiscard]] std::pair<uint32_t, uint32_t> build_cdna3_vop3(uint16_t op, uint8_t vdst,
                                                             uint16_t src0) {
  cdna3::Vop3MachineInst dst{};
  dst.encoding = 0x34;
  dst.op = op;
  dst.vdst = vdst;
  dst.src0 = src0;

  uint32_t words[2]{};
  std::memcpy(words, &dst, sizeof(dst));
  return {words[0], words[1]};
}

[[nodiscard]] std::pair<uint32_t, uint32_t> build_cdna3_vop3_sdst(uint16_t op, uint8_t sdst,
                                                                  uint8_t vdst, uint16_t src0,
                                                                  uint16_t src1,
                                                                  uint16_t src2 = 0) {
  cdna3::Vop3SdstEncMachineInst dst{};
  dst.encoding = 0x34;
  dst.op = op;
  dst.vdst = vdst;
  dst.sdst = sdst;
  dst.src0 = src0;
  dst.src1 = src1;
  dst.src2 = src2;

  uint32_t words[2]{};
  std::memcpy(words, &dst, sizeof(dst));
  return {words[0], words[1]};
}

void append_wait_all(std::vector<uint32_t> &words) { Cdna3ScratchEmitter::append_wait(words); }

[[nodiscard]] bool prepend_execz_guard(std::vector<uint32_t> &words) {
  if (words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max()))
    return false;
  words.insert(words.begin(),
               pack_sopp(cdna3::kSCbranchExeczSopp, static_cast<uint16_t>(words.size())));
  return true;
}

void append_memory_instruction(std::vector<uint32_t> &words, const Cdna3VirtualLdsAccess &access,
                               uint8_t saddr) {
  Cdna3MemoryInstructionBuilder::FlatGlobalOperands operands{};
  operands.signed_offset13 = access.byte_offset;
  operands.sc0 = true;
  operands.addr = access.address_vgpr;
  operands.saddr = saddr;
  operands.acc = access.acc;

  const auto encoded =
      access.is_load
          ? Cdna3MemoryInstructionBuilder::flat_global_load(operands, access.op, access.data_vgpr)
          : Cdna3MemoryInstructionBuilder::flat_global_store(operands, access.op, access.data_vgpr);
  words.push_back(encoded.first);
  words.push_back(encoded.second);
}

} // namespace

bool append_cdna3_virtual_lds_access(std::vector<uint32_t> &words,
                                     const TranslationContext &context,
                                     const Cdna3VirtualLdsAccess &access,
                                     std::optional<Cdna3VirtualLdsBorrowScratch> borrow_scratch) {
  if (!context.virtualize_lds || access.address_vgpr == std::numeric_limits<uint8_t>::max() ||
      (access.address_vgpr % 2) != 0 || context.virtual_lds_base_sgpr > 126 ||
      (context.virtual_lds_base_sgpr % 2) != 0) {
    return false;
  }

  const auto address_hi = static_cast<uint8_t>(access.address_vgpr + 1);
  const auto base = static_cast<uint8_t>(context.virtual_lds_base_sgpr);
  if (!context.virtual_lds_base_sgpr_spill_per_use) {
    const auto clear_high =
        build_cdna3_vop3(cdna3::kVMovB32Vop3, address_hi, scalar_positive_inline_u32(0));
    words.push_back(clear_high.first);
    words.push_back(clear_high.second);
    append_memory_instruction(words, access, base);
    append_wait_all(words);
    return true;
  }

  if (!context.virtual_lds_base_pointer_spilled || !borrow_scratch ||
      borrow_scratch->pointer_vgpr_lo == borrow_scratch->pointer_vgpr_hi ||
      borrow_scratch->saved_sgpr_private_offset >
          Cdna3ScratchEmitter::kMaxDwordOffset - sizeof(uint32_t) ||
      context.virtual_lds_base_pointer_spill_offset >
          Cdna3ScratchEmitter::kMaxDwordOffset - sizeof(uint32_t)) {
    return false;
  }

  const auto temp_lo = borrow_scratch->pointer_vgpr_lo;
  const auto temp_hi = borrow_scratch->pointer_vgpr_hi;
  if (temp_lo == access.address_vgpr || temp_lo == address_hi || temp_hi == access.address_vgpr ||
      temp_hi == address_hi) {
    return false;
  }

  std::vector<uint32_t> guarded;
  // Preserve the guest-owned scalar pair before replacing it with the backing
  // pointer saved by the entry prologue. The caller guarantees that these two
  // private slots do not overlap its own live spill payload.
  guarded.push_back(build_cdna3_v_mov_b32(temp_lo, base));
  guarded.push_back(build_cdna3_v_mov_b32(temp_hi, static_cast<uint8_t>(base + 1)));
  Cdna3ScratchEmitter::append_store_dword(guarded, temp_lo,
                                          borrow_scratch->saved_sgpr_private_offset);
  Cdna3ScratchEmitter::append_store_dword(
      guarded, temp_hi, borrow_scratch->saved_sgpr_private_offset + sizeof(uint32_t));
  Cdna3ScratchEmitter::append_load_dword(guarded, temp_lo,
                                         context.virtual_lds_base_pointer_spill_offset);
  Cdna3ScratchEmitter::append_load_dword(
      guarded, temp_hi, context.virtual_lds_base_pointer_spill_offset + sizeof(uint32_t));
  append_wait_all(guarded);
  guarded.push_back(build_cdna3_v_readfirstlane_b32(base, temp_lo));
  guarded.push_back(build_cdna3_v_readfirstlane_b32(static_cast<uint8_t>(base + 1), temp_hi));

  // In spill-per-use mode the GLOBAL instruction cannot depend on the borrowed
  // scalar pair after guest state is restored. Fold the backing pointer into an
  // even VGPR address pair and encode SADDR=null.
  const auto add_lo = build_cdna3_vop3_sdst(cdna3::kVAddCoU32Vop3SdstEnc, base, access.address_vgpr,
                                            vgpr_src(access.address_vgpr), vgpr_src(temp_lo));
  guarded.push_back(add_lo.first);
  guarded.push_back(add_lo.second);
  const auto add_hi = build_cdna3_vop3_sdst(cdna3::kVAddcCoU32Vop3SdstEnc, base, address_hi,
                                            scalar_positive_inline_u32(0), vgpr_src(temp_hi), base);
  guarded.push_back(add_hi.first);
  guarded.push_back(add_hi.second);

  append_memory_instruction(guarded, access, kCdna3ScalarNull);
  append_wait_all(guarded);

  Cdna3ScratchEmitter::append_load_dword(guarded, temp_lo,
                                         borrow_scratch->saved_sgpr_private_offset);
  Cdna3ScratchEmitter::append_load_dword(
      guarded, temp_hi, borrow_scratch->saved_sgpr_private_offset + sizeof(uint32_t));
  append_wait_all(guarded);
  guarded.push_back(build_cdna3_v_readfirstlane_b32(base, temp_lo));
  guarded.push_back(build_cdna3_v_readfirstlane_b32(static_cast<uint8_t>(base + 1), temp_hi));

  if (!prepend_execz_guard(guarded))
    return false;
  words.insert(words.end(), guarded.begin(), guarded.end());
  return true;
}

} // namespace rocjitsu
