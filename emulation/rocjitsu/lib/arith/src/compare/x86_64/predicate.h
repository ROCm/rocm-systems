// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Native x86 comparison immediates for the ISA's ordered/unordered predicates.
#pragma once

#include "common/isa.h"
#include "compare/predicate.h"

#include <immintrin.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Return the immediate implementing the requested predicate, including NaNs.
template <unsigned Predicate> constexpr int float_compare_immediate() {
  static_assert(Predicate <= CompareTrue);
  constexpr int immediates[] = {
      _CMP_FALSE_OQ, _CMP_LT_OQ,  _CMP_EQ_OQ,   _CMP_LE_OQ,   _CMP_GT_OQ, _CMP_NEQ_OQ,
      _CMP_GE_OQ,    _CMP_ORD_Q,  _CMP_UNORD_Q, _CMP_NGE_UQ,  _CMP_EQ_UQ, _CMP_NGT_UQ,
      _CMP_NLE_UQ,   _CMP_NEQ_UQ, _CMP_NLT_UQ,  _CMP_TRUE_UQ,
  };
  return immediates[Predicate];
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
