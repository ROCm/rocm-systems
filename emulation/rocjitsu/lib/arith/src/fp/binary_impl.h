// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local classification of binary arithmetic and selection operations.
#pragma once

#include "common/isa.h"
#include "fp/binary.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Identify operations that compute a result rather than select an operand.
constexpr bool is_arithmetic(Binary op) {
  return op == Binary::Add || op == Binary::Sub || op == Binary::Subrev || op == Binary::Mul ||
         op == Binary::MulDx9Zero;
}

// Identify binary selections, independently of arithmetic's supported operation set.
constexpr bool is_selection(Binary op) {
  return op == Binary::MinLegacy || op == Binary::MaxLegacy || op == Binary::MinNum ||
         op == Binary::MaxNum || op == Binary::Minimum || op == Binary::Maximum;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
