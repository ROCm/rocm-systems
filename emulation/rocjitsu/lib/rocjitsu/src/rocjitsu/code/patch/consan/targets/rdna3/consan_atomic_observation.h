// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include <array>
#include <cstring>
#include <optional>
#include <span>

namespace rocjitsu::consan::detail {
// Observe exactly one original GFX11 operation. GLC requests the overwritten
// value for atomics; it does not add an ordering operation. Store rewrites
// retain Store semantics in the journal and cannot extend a release sequence.
[[nodiscard]] inline std::optional<std::array<uint32_t, 2>>
build_rdna3_publication_observation(std::span<const uint8_t> instruction, uint16_t destination,
                                    bool store) {
  if (instruction.size() != sizeof(rdna3::FlatMachineInst) || destination > 255u)
    return std::nullopt;
  rdna3::FlatMachineInst raw{};
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  if (raw.encoding != 0x37u || (raw.seg != 0u && raw.seg != 2u) || raw.sve || raw.glc || raw.pad_25)
    return std::nullopt;
  if (raw.seg == 0u && raw.offset > 0xfffu)
    return std::nullopt;
  if (store) {
    if (raw.op != rdna3::kFlatStoreB32Flat)
      return std::nullopt;
    raw.op = rdna3::kFlatAtomicSwapB32Flat;
  } else if (raw.op != rdna3::kFlatAtomicAddU32Flat && raw.op != rdna3::kFlatAtomicOrB32Flat) {
    return std::nullopt;
  }
  raw.glc = 1u;
  raw.vdst = destination;
  std::array<uint32_t, 2> result;
  std::memcpy(result.data(), &raw, sizeof(raw));
  return result;
}
} // namespace rocjitsu::consan::detail
