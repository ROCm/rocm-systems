// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_RISC_V_ZIFENCEI_H_
#define ROCJITSU_ISA_ARCH_RISC_V_ZIFENCEI_H_

#include "rocjitsu/isa/arch/risc_v/encodings.h"

#include <cstdint>

namespace rocjitsu {
namespace risc_v {
namespace detail {

class FenceIInst : public IType {
public:
  explicit FenceIInst(uint32_t raw);
  void execute(HartState &ctx) override;
};

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_RISC_V_ZIFENCEI_H_
