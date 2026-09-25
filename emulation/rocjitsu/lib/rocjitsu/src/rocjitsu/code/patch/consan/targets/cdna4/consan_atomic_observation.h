// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include <array>
#include <cstring>
#include <optional>
#include <span>

namespace rocjitsu::consan::detail {
// The caller preserves destination in reserved VGPR scratch. SC0 requests the
// overwritten value; it does not add an ordering operation. Admit only plain
// 32-bit FLAT/GLOBAL forms, preserving their address, operand and cache fields.
[[nodiscard]] inline std::optional<std::array<uint32_t, 2>>
build_cdna4_publication_observation(std::span<const uint8_t> instruction, uint16_t destination,
                                    bool store) {
  if (instruction.size() != sizeof(cdna4::FlatGlblMachineInst) || destination > 255u)
    return std::nullopt;
  cdna4::FlatGlblMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  if (raw.encoding != 0x37u || (raw.seg != 0u && raw.seg != 2u) || raw.sve || raw.acc || raw.sc0 ||
      (raw.seg == 0u && (raw.offset & 0x1000u)))
    return std::nullopt;
  if (store) {
    if (raw.op != cdna4::kFlatStoreDwordFlat)
      return std::nullopt;
    // The journal retains Store semantics: the exchange must not extend an
    // original release sequence through this ordinary store.
    raw.op = cdna4::kFlatAtomicSwapFlat;
  } else if (raw.op != cdna4::kFlatAtomicAddFlat && raw.op != cdna4::kFlatAtomicOrFlat) {
    return std::nullopt;
  }
  raw.sc0 = 1u;
  raw.vdst = destination;
  std::array<uint32_t, 2> result;
  std::memcpy(result.data(), &raw, sizeof(raw));
  return result;
}
} // namespace rocjitsu::consan::detail
