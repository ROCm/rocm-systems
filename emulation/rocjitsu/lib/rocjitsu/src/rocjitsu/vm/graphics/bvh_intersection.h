// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu {
class Wavefront;
class ComputeUnitCore;
class VectorMemState;

bool prepare_gfx115_bvh(Wavefront &wf, VectorMemState &state, uint32_t descriptor,
                        uint32_t destination, uint32_t address, const uint32_t *nsa, bool wide,
                        bool a16, bool valid_encoding);

template <typename Mimg>
bool prepare_bvh_intersection(Wavefront &wf, VectorMemState &state, const Mimg &op,
                              const uint32_t *words, bool wide) {
  const bool valid = op.dmask == 15 && !op.d16 && op.r128 && op.unorm && !op.dim && !op.lwe &&
                     !op.tfe && !op.ssamp;
  return prepare_gfx115_bvh(wf, state, op.srsrc * 4, op.vdata, op.vaddr,
                            op.nsa ? words + 2 : nullptr, wide, op.a16, valid);
}

void complete_bvh_intersection(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &state);

} // namespace rocjitsu::amdgpu
