// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include <array>
#include <cstring>
#include <optional>
#include <span>

namespace rocjitsu::consan::detail {
// The caller reserves and preserves destination in its instrumentation scratch.
// Exactly one original RMW executes. Only VDST and the return-mode TH change;
// opcode, address, operand, offset, scope, and other cache fields are preserved.
[[nodiscard]] inline std::optional<std::array<uint32_t, 3>>
build_rdna4_atomic_observation(std::span<const uint8_t> instruction, uint16_t destination) {
  if (instruction.size() != sizeof(rdna4::VflatMachineInst) || destination > 255u)
    return std::nullopt;
  rdna4::VflatMachineInst raw{}; // FLAT and GLOBAL share this exact field layout.
  std::memcpy(&raw, instruction.data(), sizeof(raw));
  if ((raw.encoding != 0xecu && raw.encoding != 0xeeu) || raw.th != 0u || raw.pad_8_13 ||
      raw.pad_22_23 || raw.pad_40_48 || raw.pad_63 ||
      (raw.op != rdna4::kFlatAtomicAddU32Vflat && raw.op != rdna4::kFlatAtomicOrB32Vflat))
    return std::nullopt;
  raw.vdst = destination;
  raw.th = 1u; // TH_ATOMIC_RETURN; regular non-return atomics only.
  std::array<uint32_t, 3> result;
  std::memcpy(result.data(), &raw, sizeof(raw));
  return result;
}
} // namespace rocjitsu::consan::detail
