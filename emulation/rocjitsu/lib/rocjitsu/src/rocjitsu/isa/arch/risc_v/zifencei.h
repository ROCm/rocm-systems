// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/risc_v/encodings.h"

#include <cstdint>

namespace rocjitsu {
namespace risc_v {
namespace detail {

class FenceIInst : public IType {
public:
  explicit FenceIInst(uint32_t raw);
  void execute_impl(HartState &ctx);
};

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu
