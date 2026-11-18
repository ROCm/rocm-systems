// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/instruction_snapshot.h"
#include "rocjitsu/isa/register_set.h"

#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

/// A register-result dependency found before async or memory publication.
/// Buffer payload checks needing prepared lane masks follow address preparation.
/// Register numbers are wave-relative physical indices, including bank selection.
/// Lanes and bytes describe the overlap, not the whole producer or consumer.
/// XCNT replay-source dependencies are not delivered to observers.
/// Encodings are owned when available. Synthetic producers or dependencies that
/// predate observer attachment can have a PC without an encoding.
struct RegisterHazardObservation {
  uint64_t producer_pc;
  uint64_t consumer_pc;
  RegisterRef reg;
  uint64_t lane_mask;
  uint8_t byte_mask;
  bool write;
  WaitCounterKind counter;
  uint32_t required_wait;
  std::optional<InstructionSnapshot> producer_instruction;
  std::optional<InstructionSnapshot> consumer_instruction;
};

} // namespace rocjitsu::amdgpu
