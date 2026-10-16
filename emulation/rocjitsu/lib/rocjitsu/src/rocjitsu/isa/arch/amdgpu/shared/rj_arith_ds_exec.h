// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_DS_EXEC_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_DS_EXEC_H_

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>

namespace rocjitsu::amdgpu::arith {

// Original EXEC governs source participation; vector write suppression only
// filters commits. Source-lane calculations here describe observed reads, while
// rj-arith owns result computation, collision resolution, and alias-safe stores.
template <bool Backward, bool FetchInactive, typename Wave32, typename Wave64>
inline void execute_ds_permute(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                               const Operand &destination, const Operand &address,
                               const Operand &data, uint16_t offset) {
  const uint64_t exec_mask = wave.exec();
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, address, exec_mask);
  uint64_t source_mask = exec_mask;
  if constexpr (Backward) {
    if (wave.cu().observes_register_access()) {
      source_mask = 0;
      const unsigned group = ds_permute_group_width(wave.cu().arch(), wave.wf_size());
      for (unsigned lane = 0; lane < wave.wf_size(); ++lane) {
        if (!((exec_mask >> lane) & 1))
          continue;
        const unsigned source = ds_permute_lane(lane, a.data[lane], offset, group);
        if (FetchInactive || ((exec_mask >> source) & 1))
          source_mask |= 1ULL << source;
      }
    }
  }
  VectorInput32 b(registers, wave, data, source_mask);
  const uint64_t write_mask = exec_mask & wave.vgpr_write_mask();
  write_vector32_result(
      wave, registers, destination, write_mask,
      [&](uint32_t *const *d, uint64_t) {
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(exec_mask), 0, d, &a.data, &b.data, offset)
                   : wave64(flags, exec_mask, 0, d, &a.data, &b.data, offset);
      },
      /*allow_direct=*/write_mask == exec_mask);
}

// Observe only participating source lanes, including the extended rotate/FFT
// modes. Unobserved execution avoids this footprint calculation altogether.
inline void execute_ds_swizzle(Wavefront &wave, uint64_t flags, const Operand &destination,
                               const Operand &source, uint16_t offset) {
  const uint64_t exec_mask = wave.exec();
  if (!exec_mask)
    return;
  uint64_t source_mask = exec_mask;
  if (wave.cu().observes_register_access()) {
    source_mask = 0;
    for (unsigned lane = 0; lane < wave.wf_size(); ++lane) {
      if (!((exec_mask >> lane) & 1))
        continue;
      const unsigned selected = ds_swizzle_lane(lane, offset, wave.cu().arch());
      if ((exec_mask >> selected) & 1)
        source_mask |= 1ULL << selected;
    }
  }
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source, source_mask);
  const uint64_t write_mask = exec_mask & wave.vgpr_write_mask();
  write_vector32_result(
      wave, registers, destination, write_mask,
      [&](uint32_t *const *d, uint64_t) {
        return wave.wf_size() == 32
                   ? rj_arith_ds_swizzle_b32_w32(flags, uint32_t(exec_mask), 0, d, &a.data, offset)
                   : rj_arith_ds_swizzle_b32_w64(flags, exec_mask, 0, d, &a.data, offset);
      },
      /*allow_direct=*/write_mask == exec_mask);
}

} // namespace rocjitsu::amdgpu::arith

#endif // ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_DS_EXEC_H_
