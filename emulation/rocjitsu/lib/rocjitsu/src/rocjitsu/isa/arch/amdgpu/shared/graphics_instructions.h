// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_GRAPHICS_INSTRUCTIONS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_GRAPHICS_INSTRUCTIONS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <bit>

namespace rocjitsu::amdgpu {

/// GFX11+ interpolation is FMA with fixed source selection within each quad.
inline void execute_graphics_interp_f32(Wavefront &wf, uint32_t dst, std::array<uint32_t, 3> src,
                                        bool second, uint32_t neg, bool clamp, uint32_t op_sel) {
  if (!wf.exec())
    return;
  if (op_sel || dst >= wf.num_vgprs() || src[0] >= wf.num_vgprs() || src[1] >= wf.num_vgprs() ||
      src[2] >= wf.num_vgprs()) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  std::array<uint32_t, 64> result{};
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(wf.exec() & (uint64_t{1} << lane)))
      continue;
    const uint32_t quad = lane & ~3u;
    const auto read = [&](uint32_t operand, uint32_t source_lane) {
      const uint32_t value = wf.debug_read_vgpr(src[operand], source_lane);
      return std::bit_cast<float>(value ^ (((neg >> operand) & 1u) << 31));
    };
    result[lane] = fp_mode::packed_f32(read(0, quad + (second ? 2 : 1)), read(1, lane),
                                       read(2, second ? lane : quad), fp_mode::PackedF32Op::FMA,
                                       wf.fp_round_mode_f32(), wf.fp_denorm_mode_f32(), clamp, true,
                                       wf.cu().arch(), wf.ieee_mode());
  }
  // Snapshot every quad's inputs before writing: destination may alias P0/P10/P20.
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
    if (wf.exec() & (uint64_t{1} << lane))
      wf.debug_write_vgpr(dst, lane, result[lane]);
}

/// Expand one LDS parameter's P0/P10/P20 into lanes 0/1/2 of every active quad.
inline void execute_graphics_parameter_load(Wavefront &wf, uint32_t dst, uint32_t attribute,
                                            uint32_t component) {
  if (!wf.exec())
    return;
  if ((wf.m0() & 0x8000007f) || attribute > 32 || component > 3) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  if (dst >= wf.num_vgprs()) {
    wf.set_exec(0);
    return;
  }
  const uint32_t quads = wf.wf_size() / 4;
  const uint32_t starts = ((wf.m0() >> 15) | 1u) & ((1u << quads) - 1);
  const uint32_t count = std::popcount(starts);
  uint32_t primitive = 0;
  for (uint32_t quad = 0; quad < quads; ++quad) {
    if (quad && (starts & (1u << quad)))
      ++primitive;
    if (!(wf.exec() & (uint64_t{15} << (quad * 4))))
      continue;
    const uint32_t address =
        (wf.m0() & 0xffff) + 4 * (attribute * count * 12 + primitive * 12 + component * 3);
    for (uint32_t coefficient = 0; coefficient < 3; ++coefficient)
      wf.debug_write_vgpr(dst, quad * 4 + coefficient, wf.lds().read32(address + coefficient * 4));
    // The fourth lane has no parameter coefficient and is architecturally unused.
  }
}

} // namespace rocjitsu::amdgpu

#endif
