// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/instruction_compute_unit_view.h"

namespace rocjitsu::amdgpu {

/// @brief Narrow execution-only access to private compute-unit register hooks.
///
/// Only generated execution translation units include this header. Keeping the
/// access key in the VM layer prevents model-only and non-AMDGPU operand
/// subclasses from inheriting raw compute-unit access.
class OperandExecutionAccess {
public:
  static ComputeUnitCore &raw_compute_unit(InstructionComputeUnitView &view) {
    return view.raw_cu();
  }

  static const ComputeUnitCore &raw_compute_unit(const InstructionComputeUnitView &view) {
    return view.raw_cu();
  }
};

} // namespace rocjitsu::amdgpu
