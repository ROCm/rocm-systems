// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_GRAPHICS_INSTRUCTIONS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_GRAPHICS_INSTRUCTIONS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <bit>

namespace rocjitsu::amdgpu {

/// GFX11+ interpolation selects coefficient sources from fixed lanes within each quad.
inline void execute_graphics_interp(Wavefront &wf, uint32_t dst, std::array<uint32_t, 3> src,
                                    bool second, uint32_t neg, bool clamp, uint32_t op_sel,
                                    bool f16 = false, bool rtz = false) {
  if (!wf.exec())
    return;
  const uint32_t allowed_op_sel = f16 ? (1u | (1u << (second ? 3 : 2))) : 0;
  if ((op_sel & ~allowed_op_sel) || dst >= wf.num_vgprs() || src[0] >= wf.num_vgprs() ||
      src[1] >= wf.num_vgprs() || src[2] >= wf.num_vgprs()) {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  RegisterAccess regs(wf);
  const uint32_t base = wf.vgpr_alloc().base;
  std::array<std::array<float, 3>, 64> inputs{};
  std::array<uint32_t, 64> result{};
  const bool batch_registers =
      !f16 && !wf.cu().observes_register_access() && !wf.cu().debug_active();
  if (batch_registers) {
    uint64_t quads = 0;
    for (uint32_t lane = 0; lane < wf.wf_size(); lane += 4)
      if (wf.exec() & (uint64_t{15} << lane))
        quads |= uint64_t{1} << lane;
    const auto a = regs.read_vgpr_region(base + src[0], 1, quads << (second ? 2 : 1)).lanes();
    const auto b = regs.read_vgpr_region(base + src[1], 1, wf.exec()).lanes();
    const auto c = regs.read_vgpr_region(base + src[2], 1, second ? wf.exec() : quads).lanes();
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(wf.exec() & (uint64_t{1} << lane)))
        continue;
      const uint32_t quad = lane & ~3u;
      inputs[lane] = {std::bit_cast<float>(a[quad + (second ? 2 : 1)] ^ ((neg & 1u) << 31)),
                      std::bit_cast<float>(b[lane] ^ (((neg >> 1) & 1u) << 31)),
                      std::bit_cast<float>(c[second ? lane : quad] ^ (((neg >> 2) & 1u) << 31))};
    }
  } else {
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(wf.exec() & (uint64_t{1} << lane)))
        continue;
      const uint32_t quad = lane & ~3u;
      const auto read = [&](uint32_t operand, uint32_t source_lane) {
        const bool packed = f16 && (operand == 0 || (operand == 2 && !second));
        const uint32_t shift = ((op_sel >> operand) & 1u) * 16;
        uint32_t value =
            regs.read_vgpr(base + src[operand], source_lane, packed ? 3u << (shift / 8) : 15u);
        if (packed) {
          uint16_t bits = value >> shift;
          bits = fp_mode::detail::flush_input_f16(bits, wf.fp_denorm_mode_f16_f64());
          value = std::bit_cast<uint32_t>(util::f16_to_f32(bits));
        }
        return std::bit_cast<float>(value ^ (((neg >> operand) & 1u) << 31));
      };
      inputs[lane] = {read(0, quad + (second ? 2 : 1)), read(1, lane),
                      read(2, second ? lane : quad)};
    }
  }
  // Register observers retain the caller's environment. Only arithmetic shares
  // the instruction's temporary FP environment across active lanes.
  {
    const bool half_result = f16 && second;
    const uint32_t round_mode = rtz           ? 3
                                : half_result ? wf.fp_round_mode_f16_f64()
                                              : wf.fp_round_mode_f32();
    fp_mode::ScopedEnvironment environment(half_result ? 0 : round_mode);
    const bool clamp_nan = wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 || wf.dx10_clamp();
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(wf.exec() & (uint64_t{1} << lane)))
        continue;
      auto [a, b, c] = inputs[lane];
      if (!half_result) {
        result[lane] = fp_mode::detail::packed_f32_environment(
            a, b, c, fp_mode::PackedF32Op::FMA, wf.fp_denorm_mode_f32(), clamp, clamp_nan,
            wf.cu().arch(), wf.ieee_mode());
        continue;
      }
      // P2 rounds the fused result directly to F16. Rounding through an F32
      // intermediate loses directed-rounding and halfway cases.
      if (!(wf.fp_denorm_mode_f32() & 1u)) {
        for (float *value : {&b, &c}) {
          const uint32_t bits = std::bit_cast<uint32_t>(*value);
          if ((bits & 0x7f800000u) == 0)
            *value = std::bit_cast<float>(bits & 0x80000000u);
        }
      }
      double value;
      if (std::isfinite(a) && std::isfinite(b) && std::isfinite(c)) {
        const auto exact = fp_mode::detail::add_exact(double(a) * double(b), double(c));
        value = fp_mode::detail::round_to_odd(exact);
        if (exact.value == 0 && exact.error == 0) {
          const bool product_sign = std::signbit(a) != std::signbit(b);
          const bool matching_zeros = c == 0 && product_sign == std::signbit(c);
          value = (matching_zeros ? product_sign : round_mode == 2) ? -0.0 : 0.0;
        }
      } else {
        value = fp_mode::fma_f32(a, b, c, wf.cu().arch(), true, 3);
      }
      result[lane] = fp_mode::round_fma_f16(value, round_mode, wf.fp_denorm_mode_f16_f64(), clamp,
                                            wf.fp16_ovfl(), clamp_nan);
    }
  }
  // Snapshot every quad's inputs before writing: destination may alias P0/P10/P20.
  if (batch_registers) {
    const auto destination = regs.write_vgpr_region(base + dst, 1, wf.exec());
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
      if (wf.exec() & (uint64_t{1} << lane))
        destination.set_lane(0, lane, result[lane]);
    return;
  }
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(wf.exec() & (uint64_t{1} << lane)))
      continue;
    if (f16 && second) {
      const uint32_t shift = (op_sel & 8) ? 16 : 0;
      regs.write_vgpr(base + dst, lane, result[lane] << shift, 3u << (shift / 8));
    } else {
      regs.write_vgpr(base + dst, lane, result[lane]);
    }
  }
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
  RegisterAccess regs(wf);
  const uint32_t destination = wf.vgpr_alloc().base + dst;
  const uint32_t quads = wf.wf_size() / 4;
  const uint32_t starts = ((wf.m0() >> 15) | 1u) & ((1u << quads) - 1);
  const uint32_t count = std::popcount(starts);
  uint32_t primitive = 0;
  for (uint32_t quad = 0; quad < quads; ++quad) {
    if (quad && (starts & (1u << quad)))
      ++primitive;
    if (!(wf.exec() & (uint64_t{15} << (quad * 4))))
      continue;
    const uint32_t address = wf.lds_base() + (wf.m0() & 0xffff) +
                             4 * (attribute * count * 12 + primitive * 12 + component * 3);
    for (uint32_t coefficient = 0; coefficient < 3; ++coefficient)
      regs.write_vgpr(destination, quad * 4 + coefficient,
                      wf.lds().read32(address + coefficient * 4));
    // The fourth lane has no parameter coefficient and is architecturally unused.
  }
}

} // namespace rocjitsu::amdgpu

#endif
