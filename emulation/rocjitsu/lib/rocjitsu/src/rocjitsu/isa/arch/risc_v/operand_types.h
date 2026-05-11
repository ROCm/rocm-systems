// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_OPERAND_TYPES_H_
#define ROCJITSU_ISA_ARCH_RISC_V_OPERAND_TYPES_H_

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

#endif // ROCJITSU_ISA_ARCH_RISC_V_OPERAND_TYPES_H_
