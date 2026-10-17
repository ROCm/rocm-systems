// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "util/data_types.h"
#include "util/simd.h"

#include <array>
#include <bit>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

#if __has_include(<experimental/simd>)
// Each SIMD lane contains one BF16 encoding and its E8M0 exponent. Only the
// selected lanes may use floating-point fallback for subnormal BF16 inputs.
inline util::native<uint32_t> bf16_to_mxfp4_simd(util::native<uint32_t> input,
                                                 util::native<uint32_t> scale_exp,
                                                 uint64_t active_lanes) {
  using U = util::native<uint32_t>;
  using I = util::native<int32_t>;
  const U magnitude = input & U(0x7fffu);
  const U sign = (input >> 12) & U(8u);
  const I scaled = util::stdx::static_simd_cast<I>(magnitude) +
                   (I(127) - util::stdx::static_simd_cast<I>(scale_exp)) * I(128);
  I code(0);
  util::stdx::where(scaled > I(0x3e80), code) += I(1);
  util::stdx::where(scaled >= I(0x3f40), code) += I(1);
  util::stdx::where(scaled > I(0x3fa0), code) += I(1);
  util::stdx::where(scaled >= I(0x3fe0), code) += I(1);
  util::stdx::where(scaled > I(0x4020), code) += I(1);
  util::stdx::where(scaled >= I(0x4060), code) += I(1);
  util::stdx::where(scaled > I(0x40a0), code) += I(1);
  U result = sign | util::stdx::static_simd_cast<U>(code);
  util::stdx::where(magnitude == U(0), result) = sign;
  util::stdx::where(magnitude == U(0x7f80u), result) = sign | U(7u);
  util::stdx::where(magnitude > U(0x7f80u), result) = U(0);
  uint64_t subnormal =
      util::simd_mask_to_bits((magnitude > U(0)) && (magnitude < U(0x80u))) & active_lanes;
  while (subnormal) {
    const uint32_t lane = std::countr_zero(subnormal);
    subnormal &= subnormal - 1;
    result[lane] = util::bf16_to_fp4_e2m1_scaled_rne(static_cast<uint16_t>(input[lane]),
                                                     static_cast<uint8_t>(scale_exp[lane]));
  }
  return result;
}

template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cvt_scalef32_pk_fp4_bf16_simd(Inst &inst, Wavefront &wf) {
  if (util::force_scalar() || !sdwa::supports_direct_simd_store(inst) ||
      !inst.src0.simd_capable() || !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using U = util::native<uint32_t>;
  constexpr uint32_t W = U::size();
  static_assert(W <= 64 && 64 % W == 0);
  if (wf.wf_size() > 64 || wf.wf_size() % W != 0)
    return false;
  const uint64_t exec = wf.exec();
  if (!exec)
    return true;

  RegisterAccess regs(wf);
  std::array<U, 64 / W> scales;
  std::array<U, 64 / W> data;
  std::array<U, 64 / W> previous;
  const auto scale_src = regs.read_operand(inst.src1, exec);
  uint64_t data_lanes = 0;
  const uint64_t chunk_mask = util::mask<uint64_t>(W);
  for (uint32_t base = 0; base < wf.wf_size(); base += W) {
    const uint64_t chunk = (exec >> base) & chunk_mask;
    if (!chunk)
      continue;
    const U exponent = (scale_src.template load_native<uint32_t>(base) >> 23) & U(255u);
    scales[base / W] = exponent;
    data_lanes |= (util::simd_mask_to_bits(exponent != U(255u)) & chunk) << base;
  }

  // Invalid scales preserve the destination without reading the data operand.
  // Snapshot all inputs before acquiring writes, including when the destination
  // aliases data/scale or its backing is still an unmaterialized logical zero.
  std::optional<RegisterAccess::OperandReadView> data_src;
  if (data_lanes)
    data_src.emplace(regs.read_operand(inst.src0, data_lanes));
  const auto old_dst = regs.read_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += W) {
    if (!((exec >> base) & chunk_mask))
      continue;
    previous[base / W] = old_dst.template load_native<uint32_t>(base);
    if ((data_lanes >> base) & chunk_mask)
      data[base / W] = data_src->template load_native<uint32_t>(base);
  }

  // The scalar instruction observes full-word reads and writes even though it
  // replaces one byte. Preserve that footprint, including invalid-scale writes.
  const auto dst = regs.write_operand(inst.vdst, exec);
  const uint32_t shift = ((inst.inst_.op_sel >> 2) & 3u) * 8;
  for (uint32_t base = 0; base < wf.wf_size(); base += W) {
    const uint64_t chunk = (exec >> base) & chunk_mask;
    if (!chunk)
      continue;
    U result = previous[base / W];
    const uint64_t valid = (data_lanes >> base) & chunk_mask;
    if (valid) {
      const U input = data[base / W];
      const U exponent = scales[base / W];
      const U packed = bf16_to_mxfp4_simd(input & U(0xffffu), exponent, valid) |
                       (bf16_to_mxfp4_simd(input >> 16, exponent, valid) << 4);
      util::stdx::where(util::simd_mask_from_bits<U>(valid), result) =
          (result & U(~(255u << shift))) | (packed << shift);
    }
    dst.template store_native<uint32_t>(base, result, chunk);
  }
  return true;
}
#endif

template <typename Inst>
[[nodiscard]] bool try_execute_cvt_scalef32_pk_fp4_bf16_simd(Inst &, Wavefront &) {
  return false;
}

} // namespace rocjitsu::amdgpu
