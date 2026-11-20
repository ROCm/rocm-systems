// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/image_address.h"
#include "util/result.h"
#include <cstdint>

namespace rocjitsu::amdgpu {
class Wavefront;
class ComputeUnitCore;
class VectorMemState;

util::Result prepare_bvh(Wavefront &wf, VectorMemState &state, uint32_t descriptor,
                         uint32_t destination, const ImageAddressRegisters &registers, BvhForm form,
                         bool a16, bool valid_encoding);

template <typename Vimage>
util::Result prepare_gfx12_bvh(Wavefront &wf, VectorMemState &state, const Vimage &op,
                               BvhForm form) {
  const auto bases = gfx12_image_address_registers(op);
  const auto widths = bvh_group_widths(form, op.a16);
  ImageAddressRegisters registers{};
  unsigned word = 0;
  for (unsigned group = 0; group < widths.size(); ++group)
    for (unsigned i = 0; i < widths[group]; ++i)
      registers[word++] = bases[group] + i;
  // UNORM, LWE and SSAMP are absent from VIMAGE.
  const bool valid = op.dmask == 15 && op.r128 && !op.dim && !op.d16 && !op.tfe &&
                     !(bvh_two_results(form) && op.a16);
  return prepare_bvh(wf, state, op.rsrc, op.vdata, registers, form, op.a16, valid);
}

/// Require DMASK=15, R128 and UNORM, with zero D16, DIM, LWE, TFE and SSAMP.
template <typename Mimg>
util::Result prepare_bvh_intersection(Wavefront &wf, VectorMemState &state, const Mimg &op,
                                      const uint32_t *words, bool wide) {
  const bool valid = op.dmask == 15 && !op.d16 && op.r128 && op.unorm && !op.dim && !op.lwe &&
                     !op.tfe && !op.ssamp;
  const auto registers = gfx11_bvh_registers(wide, op.a16, op.vaddr, op.nsa ? words + 2 : nullptr);
  return prepare_bvh(wf, state, op.srsrc * 4, op.vdata, registers,
                     wide ? BvhForm::Bvh64 : BvhForm::Bvh, op.a16, valid);
}

/// Publish the completed intersection to the active lanes.
void complete_bvh_intersection(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &state);

} // namespace rocjitsu::amdgpu
