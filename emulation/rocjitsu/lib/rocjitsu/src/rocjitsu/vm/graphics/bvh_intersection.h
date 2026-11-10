// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/result.h"
#include <cstdint>

namespace rocjitsu::amdgpu {
class Wavefront;
class ComputeUnitCore;
class VectorMemState;

/// @brief Prepare a GFX11 intersection and capture its ray operands.
/// @details descriptor is an already-scaled SGPR selector. Destination and address
/// are wave-relative VGPR indices. NSA points to encoding word 2 or is null.
util::Result prepare_gfx11_bvh_intersection(Wavefront &wf, VectorMemState &state,
                                            uint32_t descriptor, uint32_t destination,
                                            uint32_t address, const uint32_t *nsa, bool wide,
                                            bool a16, bool valid_encoding);

/// Require DMASK=15, R128 and UNORM, with zero D16, DIM, LWE, TFE and SSAMP.
template <typename Mimg>
util::Result prepare_bvh_intersection(Wavefront &wf, VectorMemState &state, const Mimg &op,
                                      const uint32_t *words, bool wide) {
  const bool valid = op.dmask == 15 && !op.d16 && op.r128 && op.unorm && !op.dim && !op.lwe &&
                     !op.tfe && !op.ssamp;
  return prepare_gfx11_bvh_intersection(wf, state, op.srsrc * 4, op.vdata, op.vaddr,
                                        op.nsa ? words + 2 : nullptr, wide, op.a16, valid);
}

/// Write all four result words or none if their full VGPR tuple is unallocated.
void complete_bvh_intersection(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &state);

} // namespace rocjitsu::amdgpu
