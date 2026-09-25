// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include <array>
#include <cstring>
#include <optional>
#include <span>

namespace rocjitsu::consan::detail {
// Observe exactly one original operation. CDNA5's scale_offset is an address
// field, not RDNA4's reserved bit: preserve it along with scope and operands.
// Store rewrites retain Store semantics in the journal and cannot extend a
// release sequence merely because the observation uses an exchange.
[[nodiscard]] inline std::optional<std::array<uint32_t, 3>>
build_cdna5_publication_observation(std::span<const uint8_t> instruction, uint16_t destination,
                                    bool store) {
  if (instruction.size() != sizeof(cdna5::VflatMachineInst) || destination > 255u)
    return std::nullopt;
  cdna5::VflatMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  if ((raw.encoding != 0xecu && raw.encoding != 0xeeu) || raw.th != 0u || raw.pad_8_13 ||
      raw.pad_22_23 || raw.pad_40_47 || raw.pad_63)
    return std::nullopt;
  if (store) {
    if (raw.op != cdna5::kFlatStoreB32Vflat)
      return std::nullopt;
    raw.op = cdna5::kFlatAtomicSwapB32Vflat;
  } else if (raw.op != cdna5::kFlatAtomicAddU32Vflat && raw.op != cdna5::kFlatAtomicOrB32Vflat) {
    return std::nullopt;
  }
  raw.vdst = destination;
  raw.th = 1u;
  std::array<uint32_t, 3> result;
  std::memcpy(result.data(), &raw, sizeof(raw));
  return result;
}
} // namespace rocjitsu::consan::detail
