// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic_translator.cpp
/// @brief Semantic translator implementation and per-pair rule tables.

#include "rocjitsu/code/dbt/semantic_translator.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <string_view>

namespace rocjitsu {

// --- Waitcnt decode/encode ---

namespace {

[[nodiscard]] uint32_t make_gfx12_sopp(uint8_t op, uint16_t simm16) {
  rdna4::SoppMachineInst s{};
  s.encoding = 0x17F;
  s.op = op;
  s.simm16 = simm16;
  return std::bit_cast<uint32_t>(s);
}

constexpr uint8_t kOpWaitLoadcnt = 64;
constexpr uint8_t kOpWaitStorecntDscnt = 73;
constexpr uint8_t kOpWaitKmcnt = 71;
constexpr uint8_t kOpWaitExpcnt = 68;

} // namespace

WaitcntValues decode_waitcnt_gfx9(uint16_t simm16) {
  WaitcntValues v;
  v.vmcnt = (simm16 & 0xF) | (static_cast<uint8_t>((simm16 >> 14) & 0x3) << 4);
  v.expcnt = static_cast<uint8_t>((simm16 >> 4) & 0x7);
  v.lgkmcnt = static_cast<uint8_t>((simm16 >> 8) & 0x0F);
  return v;
}

std::vector<uint32_t> encode_waitcnt_gfx12(const WaitcntValues &vals) {
  std::vector<uint32_t> words;

  const bool need_loadcnt = (vals.vmcnt != 0x3F);
  const bool need_storecnt = (vals.vmcnt != 0x3F);
  const bool need_kmcnt = (vals.lgkmcnt != 0x0F);
  const bool need_dscnt = (vals.lgkmcnt != 0x0F);
  const bool need_expcnt = (vals.expcnt != 0x07);

  if (need_loadcnt)
    words.push_back(make_gfx12_sopp(kOpWaitLoadcnt, std::min<uint16_t>(vals.vmcnt, 63)));

  if (need_storecnt || need_dscnt) {
    const uint8_t sc = need_storecnt ? std::min<uint8_t>(vals.vmcnt, 15) : 15;
    const uint8_t dc = need_dscnt ? std::min<uint8_t>(vals.lgkmcnt, 15) : 15;
    words.push_back(make_gfx12_sopp(kOpWaitStorecntDscnt, (sc << 4) | dc));
  }

  if (need_kmcnt)
    words.push_back(make_gfx12_sopp(kOpWaitKmcnt, std::min<uint16_t>(vals.lgkmcnt, 15)));

  if (need_expcnt)
    words.push_back(make_gfx12_sopp(kOpWaitExpcnt, vals.expcnt));

  if (words.empty())
    words.push_back(build_s_nop());

  return words;
}

// --- Semantic rules ---

namespace {

SemanticReplacement translate_waitcnt_gfx9_to_gfx12(const Instruction &inst, uint64_t offset,
                                                    rj_code_arch_t) {
  if (!inst.raw_encoding())
    return {};

  const auto &sopp = *reinterpret_cast<const cdna4::SoppMachineInst *>(inst.raw_encoding());
  auto vals = decode_waitcnt_gfx9(sopp.simm16);
  auto words = encode_waitcnt_gfx12(vals);
  return {offset, offset + inst.size(), std::move(words)};
}

/// @brief Semantic rule table for CDNA4 → RDNA4 translation.
constexpr SemanticRule kRules_cdna4_to_rdna4[] = {
    {"waitcnt_gfx9_to_gfx12", WAITCNT, translate_waitcnt_gfx9_to_gfx12},
};

} // namespace

SemanticTranslator::SemanticTranslator(rj_code_arch_t guest, rj_code_arch_t host)
    : guest_arch_(guest), host_arch_(host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    rules_ = kRules_cdna4_to_rdna4;
}

std::vector<SemanticReplacement> SemanticTranslator::translate(BasicBlock &block) const {
  std::vector<SemanticReplacement> result;
  uint64_t offset = block.start_offset();
  for (auto it = block.instructions().begin(); it != block.instructions().end(); ++it) {
    const auto &inst = *it;
    for (const auto &rule : rules_) {
      if (!(inst.flags() & rule.anchor_flags))
        continue;
      auto repl = rule.translate(inst, offset, host_arch_);
      if (repl.matched()) {
        result.push_back(std::move(repl));
        break;
      }
    }
    offset += inst.size();
  }
  return result;
}

// --- Instruction lowering (Action::Expand) ---

namespace {

std::vector<uint32_t> lower_v_lshl_add_u64(const Instruction &inst,
                                           [[maybe_unused]] rj_code_arch_t host_arch) {
  const auto *raw = inst.raw_encoding();
  if (!raw || inst.size() < 8)
    return {};

  cdna4::Vop3MachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  const uint16_t vdst = src.vdst;
  const uint16_t src0 = src.src0;
  const uint16_t src2 = src.src2;

  constexpr uint16_t VCC_LO = 106;
  std::vector<uint32_t> words;

  // v_add_co_u32 vdst_lo, vcc_lo, src0_lo, src2_lo
  {
    const uint32_t w0 = (0x35u << 26) | (768u << 16) | (VCC_LO << 8) | vdst;
    const uint32_t w1 = static_cast<uint32_t>(src0) | (static_cast<uint32_t>(src2) << 9);
    words.push_back(w0);
    words.push_back(w1);
  }

  // s_wait_alu 0xFFFD
  words.push_back(pack_sopp(8, 0xFFFD));

  // v_add_co_ci_u32 vdst_hi, vcc_lo, src0_hi, src2_hi, vcc_lo
  {
    const uint32_t w0 = (0x35u << 26) | (288u << 16) | (VCC_LO << 8) | (vdst + 1);
    const uint32_t w1 = static_cast<uint32_t>(src0 + 1) | (static_cast<uint32_t>(src2 + 1) << 9) |
                        (static_cast<uint32_t>(VCC_LO) << 18);
    words.push_back(w0);
    words.push_back(w1);
  }

  return words;
}

} // namespace

std::vector<uint32_t> SemanticTranslator::try_lower_expand(const Instruction &inst) const {
  if (std::string_view(inst.mnemonic()) == "v_lshl_add_u64")
    return lower_v_lshl_add_u64(inst, host_arch_);
  return {};
}

// --- Workgroup ID rewrite ---

std::vector<SemanticReplacement> SemanticTranslator::rewrite_workgroup_ids(
    BasicBlock &block, std::span<const CodeObjectPatcher::WorkGroupIdInfo> wg_info,
    std::span<const uint8_t> translated_text) const {
  if (host_arch_ != ROCJITSU_CODE_ARCH_RDNA4 || wg_info.empty())
    return {};

  // RDNA4 delivers workgroup IDs via TTMP registers, not SGPRs.
  constexpr uint8_t kTTMP9 = 117;                  // workgroup_id_x
  [[maybe_unused]] constexpr uint8_t kTTMP7 = 115; // workgroup_id_y (low16), _z (high16)

  std::vector<SemanticReplacement> result;

  for (const auto &info : wg_info) {
    if (info.sgpr_wg_id_x < 0)
      continue;
    const auto old_sgpr = static_cast<uint16_t>(info.sgpr_wg_id_x);

    uint64_t offset = block.start_offset();
    for (const auto &inst : block.instructions()) {
      if (offset < info.entry_text_offset || offset + 8 > translated_text.size()) {
        offset += inst.size();
        continue;
      }

      if (inst.size() < 8) {
        offset += inst.size();
        continue;
      }

      // Read the already-translated instruction words from translated_text.
      uint32_t w0, w1;
      std::memcpy(&w0, translated_text.data() + offset, 4);
      std::memcpy(&w1, translated_text.data() + offset + 4, 4);

      // VOP3 word1 has src0 in bits[8:0]. Check if it matches the
      // workgroup_id SGPR and substitute TTMP9.
      uint16_t src0 = w1 & 0x1FF;
      if (src0 == old_sgpr) {
        uint32_t new_w1 = (w1 & ~0x1FFu) | kTTMP9;
        result.push_back({offset, offset + static_cast<uint64_t>(inst.size()), {w0, new_w1}});
      }

      offset += inst.size();
    }
  }
  return result;
}

} // namespace rocjitsu
