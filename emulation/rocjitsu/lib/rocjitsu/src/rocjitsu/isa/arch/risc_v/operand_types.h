// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocjitsu {
namespace risc_v {
namespace detail {

enum class OperandType {
  OPR_GPR, // x0-x31 (integer general purpose registers)
  OPR_FPR, // f0-f31 (floating-point registers)
  OPR_IMM, // immediate value
  OPR_CSR, // control/status register address
};

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu
