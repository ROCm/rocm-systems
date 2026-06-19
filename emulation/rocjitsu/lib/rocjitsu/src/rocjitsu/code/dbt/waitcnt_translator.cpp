// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcnt_translator.cpp
/// @brief Wait-counter decode and target split-wait encoding helpers.

#include "rocjitsu/code/dbt/waitcnt_translator.h"

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"

#include <algorithm>
#include <bit>

namespace rocjitsu {
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
  // GFX9 vmcnt is a unified counter for both loads and stores. GFX12 splits
  // the queues, so emit storecnt whenever vmcnt is active; the source encoding
  // does not tell us whether the waited vector memory op was a load or store.
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

} // namespace rocjitsu
