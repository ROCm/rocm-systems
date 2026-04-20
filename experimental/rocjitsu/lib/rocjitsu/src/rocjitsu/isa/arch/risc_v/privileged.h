// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_PRIVILEGED_H_
#define ROCJITSU_ISA_ARCH_RISC_V_PRIVILEGED_H_

#include "rocjitsu/isa/arch/risc_v/encodings.h"
#include "rocjitsu/isa/arch/risc_v/operand.h"

#include <cstdint>

namespace rocjitsu {
namespace risc_v {
namespace detail {

class SretInst : public RType {
public:
  explicit SretInst(uint32_t raw);
  void execute_impl(HartState &ctx);
};

class MretInst : public RType {
public:
  explicit MretInst(uint32_t raw);
  void execute_impl(HartState &ctx);
};

class WfiInst : public RType {
public:
  explicit WfiInst(uint32_t raw);
  void execute_impl(HartState &ctx);
};

class SfenceVmaInst : public RType {
public:
  explicit SfenceVmaInst(uint32_t raw);
  void execute_impl(HartState &ctx);

private:
  Operand rs1;
  Operand rs2;
};

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_RISC_V_PRIVILEGED_H_
