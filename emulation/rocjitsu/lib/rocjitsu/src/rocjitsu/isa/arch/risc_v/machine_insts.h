// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_MACHINE_INSTS_H_
#define ROCJITSU_ISA_ARCH_RISC_V_MACHINE_INSTS_H_

#include <cstdint>

namespace rocjitsu {
namespace risc_v {
namespace detail {

using MachineInst = uint32_t;

struct RTypeMachineInst {
  uint32_t opcode : 7;
  uint32_t rd : 5;
  uint32_t funct3 : 3;
  uint32_t rs1 : 5;
  uint32_t rs2 : 5;
  uint32_t funct7 : 7;
};

struct ITypeMachineInst {
  uint32_t opcode : 7;
  uint32_t rd : 5;
  uint32_t funct3 : 3;
  uint32_t rs1 : 5;
  uint32_t imm11_0 : 12;
};

struct STypeMachineInst {
  uint32_t opcode : 7;
  uint32_t imm4_0 : 5;
  uint32_t funct3 : 3;
  uint32_t rs1 : 5;
  uint32_t rs2 : 5;
  uint32_t imm11_5 : 7;
};

struct BTypeMachineInst {
  uint32_t opcode : 7;
  uint32_t imm11 : 1;
  uint32_t imm4_1 : 4;
  uint32_t funct3 : 3;
  uint32_t rs1 : 5;
  uint32_t rs2 : 5;
  uint32_t imm10_5 : 6;
  uint32_t imm12 : 1;
};

struct UTypeMachineInst {
  uint32_t opcode : 7;
  uint32_t rd : 5;
  uint32_t imm31_12 : 20;
};

struct JTypeMachineInst {
  uint32_t opcode : 7;
  uint32_t rd : 5;
  uint32_t imm19_12 : 8;
  uint32_t imm11 : 1;
  uint32_t imm10_1 : 10;
  uint32_t imm20 : 1;
};

struct R4TypeMachineInst {
  uint32_t opcode : 7;
  uint32_t rd : 5;
  uint32_t funct3 : 3;
  uint32_t rs1 : 5;
  uint32_t rs2 : 5;
  uint32_t fmt : 2;
  uint32_t rs3 : 5;
};

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_RISC_V_MACHINE_INSTS_H_
