// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "atomic_fixture.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"

#include <array>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::test {

inline constexpr std::array<AtomicTargetCase, 2> kNativeCdnaExchangeCases = {{
    {.target = ROCJITSU_CODE_TARGET_GFX942,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX942,
     .arch = IsaFamily::Cdna3,
     .wave = 64,
     .name = "Gfx942Wave64"},
    {.target = ROCJITSU_CODE_TARGET_GFX950,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX950,
     .arch = IsaFamily::Cdna4,
     .wave = 64,
     .name = "Gfx950Wave64"},
}};

enum class NativeCdnaExchangeForm : uint16_t {
  ExchangeB32 = cdna3::kDsWrxchgRtnB32Ds,
  Exchange2B32 = cdna3::kDsWrxchg2RtnB32Ds,
  Exchange2Stride64B32 = cdna3::kDsWrxchg2st64RtnB32Ds,
  ExchangeB64 = cdna3::kDsWrxchgRtnB64Ds,
  Exchange2B64 = cdna3::kDsWrxchg2RtnB64Ds,
  Exchange2Stride64B64 = cdna3::kDsWrxchg2st64RtnB64Ds,
  WideAddB64 = cdna3::kDsAddU64Ds,
};

static_assert(cdna3::kDsWrxchgRtnB32Ds == cdna4::kDsWrxchgRtnB32Ds &&
              cdna3::kDsWrxchg2RtnB32Ds == cdna4::kDsWrxchg2RtnB32Ds &&
              cdna3::kDsWrxchg2st64RtnB32Ds == cdna4::kDsWrxchg2st64RtnB32Ds &&
              cdna3::kDsWrxchgRtnB64Ds == cdna4::kDsWrxchgRtnB64Ds &&
              cdna3::kDsWrxchg2RtnB64Ds == cdna4::kDsWrxchg2RtnB64Ds &&
              cdna3::kDsWrxchg2st64RtnB64Ds == cdna4::kDsWrxchg2st64RtnB64Ds &&
              cdna3::kDsAddU64Ds == cdna4::kDsAddU64Ds);

inline constexpr std::array<NativeCdnaExchangeForm, 7> kNativeCdnaExchangeInventoryForms = {
    NativeCdnaExchangeForm::ExchangeB64,  NativeCdnaExchangeForm::ExchangeB32,
    NativeCdnaExchangeForm::Exchange2B32, NativeCdnaExchangeForm::Exchange2Stride64B32,
    NativeCdnaExchangeForm::Exchange2B64, NativeCdnaExchangeForm::Exchange2Stride64B64,
    NativeCdnaExchangeForm::WideAddB64};

inline std::string_view native_cdna_exchange_mnemonic(NativeCdnaExchangeForm form) {
  switch (form) {
  case NativeCdnaExchangeForm::ExchangeB32:
    return "ds_wrxchg_rtn_b32";
  case NativeCdnaExchangeForm::Exchange2B32:
    return "ds_wrxchg2_rtn_b32";
  case NativeCdnaExchangeForm::Exchange2Stride64B32:
    return "ds_wrxchg2st64_rtn_b32";
  case NativeCdnaExchangeForm::ExchangeB64:
    return "ds_wrxchg_rtn_b64";
  case NativeCdnaExchangeForm::Exchange2B64:
    return "ds_wrxchg2_rtn_b64";
  case NativeCdnaExchangeForm::Exchange2Stride64B64:
    return "ds_wrxchg2st64_rtn_b64";
  case NativeCdnaExchangeForm::WideAddB64:
    return "ds_add_u64";
  }
  return {};
}

inline std::array<uint32_t, 2> native_cdna_exchange_words(const AtomicTargetCase &target,
                                                          NativeCdnaExchangeForm form,
                                                          bool modifiers = false) {
  const bool wide = form == NativeCdnaExchangeForm::ExchangeB64 ||
                    form == NativeCdnaExchangeForm::Exchange2B64 ||
                    form == NativeCdnaExchangeForm::Exchange2Stride64B64 ||
                    form == NativeCdnaExchangeForm::WideAddB64;
  const uint8_t offset0 = wide ? 8 : 4;
  const uint8_t flag = static_cast<uint8_t>(modifiers);
  const uint16_t opcode = static_cast<uint16_t>(form);
  if (target.arch == IsaFamily::Cdna3)
    return cdna3::build_ds(opcode, {.offset0 = offset0,
                                    .offset1 = 5,
                                    .gds = flag,
                                    .acc = flag,
                                    .addr = 2,
                                    .data0 = 4,
                                    .data1 = 6,
                                    .vdst = 8});
  return cdna4::build_ds(opcode, {.offset0 = offset0,
                                  .offset1 = 5,
                                  .gds = flag,
                                  .acc = flag,
                                  .addr = 2,
                                  .data0 = 4,
                                  .data1 = 6,
                                  .vdst = 8});
}

inline std::vector<uint8_t> make_native_cdna_atomic_sequence_elf(const AtomicTargetCase &target,
                                                                 std::vector<uint32_t> words) {
  words.push_back(target.arch == IsaFamily::Cdna3 ? cdna3::build_sopp(cdna3::kSWaitcntSopp)[0]
                                                  : cdna4::build_sopp(cdna4::kSWaitcntSopp)[0]);
  words.push_back(build_s_endpgm(target.arch).value());
  std::vector<uint8_t> image =
      make_amdgpu_kernel_elf(words, 0, 3, target.machine, 15, 7, false, false, false, false, false);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}

inline std::vector<uint8_t>
make_native_cdna_exchange_inventory_elf(const AtomicTargetCase &target) {
  std::vector<uint32_t> words;
  for (NativeCdnaExchangeForm form : kNativeCdnaExchangeInventoryForms) {
    const std::array<uint32_t, 2> instruction = native_cdna_exchange_words(target, form);
    words.insert(words.end(), instruction.begin(), instruction.end());
  }
  return make_native_cdna_atomic_sequence_elf(target, std::move(words));
}

} // namespace rocjitsu::test
