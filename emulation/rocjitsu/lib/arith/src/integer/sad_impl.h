// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local SAD layout and masking policy queries.
#pragma once

#include "common/isa.h"
#include "integer/sad.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Return whether the SAD operation computes a quad of overlapping differences.
constexpr bool sad_quad(Sad op) { return op >= Sad::QuadU16; }

// Return whether SAD results are packed into 16-bit halves.
constexpr bool sad_packed(Sad op) { return op == Sad::QuadU16 || op == Sad::MaskedQuadU16; }

// Return whether zero source bytes suppress their SAD contributions.
constexpr bool sad_masked(Sad op) {
  return op == Sad::MaskedU8 || op == Sad::MaskedQuadU16 || op == Sad::MaskedQuadU32;
}

// Return the number of destination VGPRs required by this SAD operation.
constexpr int sad_outputs(Sad op) { return op == Sad::MaskedQuadU32 ? 4 : sad_quad(op) ? 2 : 1; }

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
