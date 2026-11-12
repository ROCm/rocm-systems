// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>
#include <format>

namespace rocjitsu {
namespace cdna4 {

std::optional<uint64_t> smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf,
                                               amdgpu::ScalarMemState *state) {
  return amdgpu::addr_calc::smem_calculate_address(inst, wf, state, /*check_negative_offset=*/true);
}

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::flat_calculate_addresses(inst, wf, d);
}

void mubuf_calculate_addresses(const MubufMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::mubuf_calculate_addresses(inst, wf, d);
}

void mtbuf_calculate_addresses(const MtbufMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::mtbuf_calculate_addresses(inst, wf, d);
}

void ds_calculate_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                            amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::ds_calculate_addresses(inst, wf, d);
  const auto transpose = static_cast<amdgpu::TransposeKind>(d.transpose);
  const bool b64 = transpose == amdgpu::TransposeKind::CDNA4_TR_B4 ||
                   transpose == amdgpu::TransposeKind::B64_TR_B8 ||
                   transpose == amdgpu::TransposeKind::B64_TR_B16;
  if ((!b64 && transpose != amdgpu::TransposeKind::CDNA4_TR_B6) || !d.exec_mask)
    return;

  // CDNA4 section 11.4 requires all lanes for these cross-lane loads. Empty
  // EXEC issues no requests, as for the existing global-transpose diagnostic.
  const uint64_t full_exec = ~uint64_t{0} >> (64 - wf.wf_size());
  if (d.exec_mask != full_exec)
    wf.report_undefined_behavior("LDS transpose load requires a full or empty EXEC mask");

  // The B64 forms require eight-byte addresses, including the immediate offset.
  // The manual does not specify a numerical alignment for the packed B96 form;
  // do not infer twelve- or sixteen-byte alignment from its payload width.
  if (!b64)
    return;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(d.exec_mask & (uint64_t{1} << lane)))
      continue;
    const uint64_t address = d.per_lane_addr[lane] - wf.lds_base();
    if (address & 7) {
      wf.report_undefined_behavior(std::format(
          "LDS transpose load requires an 8-byte-aligned address (lane {} address={:#x})", lane,
          address));
      break;
    }
  }
}

} // namespace cdna4
} // namespace rocjitsu
